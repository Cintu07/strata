"""capturing a real router trace from a hugging face moe model.

this is the only module that needs torch and transformers, and it is imported
lazily so that the analysis half of the harness runs without them.

# how it works

moe implementations differ, but they all have a router: a small module, one per
block, that maps the hidden state to per-expert scores and picks the top-k. a
forward hook on it sees both the input it was given and the decision it made,
which is exactly the pair the speculative router head would have to learn.

# the two things that make this harder than it sounds

**a router is not always a `Linear`.** granitemoe's `GraniteMoeTopKRouter` holds
a bare weight and calls `F.linear` itself, so a search for `nn.Linear` modules
finds the attention projections and no routers at all. discovery therefore keys
on the module's position and class name, and then *verifies* that it can produce
an output of the right width.

**a router does not always return logits.** mixtral's gate returns
`(num_tokens, num_experts)` logits. granitemoe's returns a three tuple of
`(top_k_index, top_k_weights, router_logits)`, so taking element zero and running
top-k over it treats expert *indices* as scores and records confident nonsense.
that failure produces a complete, well shaped, entirely fictional trace, which is
the worst kind. so the extraction identifies tensors by dtype and width rather
than by position, and prefers the router's own choice when it offers one.

# hidden states are large

a hidden state is one vector per token per layer, and at four thousand dimensions
across sixty layers that is a megabyte per token. the probe does not need full
resolution, so the capture projects down to `probe_dim` with a fixed random
projection. a random projection preserves the geometry the probe cares about
(johnson lindenstrauss), it is seeded so the capture is reproducible, and it
turns an unusable file into a few hundred megabytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .trace import SOURCE_CAPTURED, Provenance, RouterTrace


@dataclass
class CaptureConfig:
    """what to capture and how much of it."""

    model_id: str
    corpus: str = ""
    max_tokens: int = 4096
    top_k: int | None = None
    """expert count per token. read from the model config when left as None."""
    probe_dim: int = 128
    """dimension the hidden state is projected to. set to 0 to skip hidden
    states entirely, which makes the file small and the prediction analysis
    impossible."""
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    max_memory: str | None = None
    """ram ceiling for the weights, e.g. ``"9GiB"``. when set, the model is
    dispatched layer by layer and whatever does not fit is streamed from
    ``offload_dir``.

    this exists because the models worth measuring are the ones that do not fit.
    olmoe-1b-7b is 13.8gb of weights against 15.6gb of host ram, and a plain
    ``from_pretrained`` followed by ``.to(device)`` needs the whole thing
    resident at once plus room for activations, so it dies after the download
    rather than before it. m0 only needs router logits and a hidden state, so
    there is no reason to hold the experts in memory at all.
    """

    offload_dir: str = "offload"
    """where dispatched weights are streamed from. ignored unless
    ``max_memory`` is set. put it on the fastest disk available, because the
    forward pass reads through it."""

    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)


def extract_routing(output: Any, n_experts: int, top_k: int) -> tuple[str, Any]:
    """pull the routing decision out of whatever the router returned.

    identifies tensors by dtype and width rather than by position in the return
    value, because that position differs between families and getting it wrong
    is silent.

    returns `("chosen", indices)` when the router hands over its own top-k, which
    is preferred because it is authoritative and cannot disagree with the model
    over a tie. otherwise `("logits", scores)` and the caller takes the top-k.

    raises if neither is present, rather than returning something plausible.
    """
    import torch

    tensors = output if isinstance(output, (tuple, list)) else [output]
    tensors = [t for t in tensors if torch.is_tensor(t)]

    for t in tensors:
        if not t.is_floating_point() and t.shape[-1] == top_k:
            return "chosen", t
    for t in tensors:
        if t.is_floating_point() and t.shape[-1] == n_experts:
            return "logits", t

    shapes = [(tuple(t.shape), str(t.dtype)) for t in tensors]
    raise RuntimeError(
        f"a router returned nothing that looks like a routing decision. "
        f"expected an integer tensor of width {top_k} or a float tensor of "
        f"width {n_experts}, got {shapes}"
    )


class RouterCapture:
    """hooks a model's routers and accumulates a trace.

    usage is deliberately two steps, so the caller keeps control of how the
    model is loaded and what is run through it:

    ```python
    capture = RouterCapture(model, config)
    with capture:
        for batch in corpus:
            model(**batch)
    trace = capture.finish()
    ```
    """

    def __init__(self, model: Any, config: CaptureConfig) -> None:
        self.model = model
        self.config = config
        self._handles: list[Any] = []
        self._routing: list[np.ndarray] = []
        self._hidden: list[np.ndarray] = []
        self._projection: np.ndarray | None = None
        self._pending: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        self._segments: list[int] = []
        self.n_experts = infer_n_experts(model)
        self.top_k = config.top_k or infer_top_k(model)
        self.routers = find_routers(model, self.n_experts)

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "RouterCapture":
        for index, (_, module) in enumerate(self.routers):
            self._handles.append(module.register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_index: int):
        import torch

        def hook(_module: Any, inputs: tuple, output: Any) -> None:
            with torch.no_grad():
                hidden = inputs[0].detach()
                hidden = hidden.reshape(-1, hidden.shape[-1]).float().cpu().numpy()

                kind, tensor = extract_routing(output, self.n_experts, self.top_k)
                tensor = tensor.detach().reshape(-1, tensor.shape[-1]).cpu()
                if kind == "chosen":
                    # the router's own decision, which cannot disagree with the
                    # model about a tie
                    chosen = tensor.numpy()
                else:
                    scores = tensor.float().numpy()
                    chosen = np.argpartition(-scores, self.top_k - 1, axis=1)[:, : self.top_k]

                projected = None
                if self.config.probe_dim > 0:
                    projected = self._project(hidden)
                self._pending[layer_index] = (chosen.astype(np.int32), projected)

                # a full pass through every layer completes one block of tokens
                if len(self._pending) == len(self.routers):
                    self._flush()

        return hook

    def _project(self, hidden: np.ndarray) -> np.ndarray:
        if self._projection is None:
            rng = np.random.default_rng(self.config.seed)
            d = hidden.shape[1]
            self._projection = (
                rng.normal(size=(d, self.config.probe_dim)) / np.sqrt(d)
            ).astype(np.float32)
        return hidden @ self._projection

    def _flush(self) -> None:
        order = sorted(self._pending)
        routing = np.stack([self._pending[i][0] for i in order], axis=1)
        self._routing.append(routing)
        if self.config.probe_dim > 0:
            hidden = np.stack([self._pending[i][1] for i in order], axis=1)
            self._hidden.append(hidden.astype(np.float32))
        self._pending.clear()

    def mark_segment(self) -> None:
        """record that a new stretch of text starts at the next token.

        called once per subject in the corpus, so the analysis can ask whether
        routing depends on what is being talked about. cheap to record and
        impossible to recover afterwards.
        """
        at = self.tokens_captured
        if not self._segments or self._segments[-1] != at:
            self._segments.append(at)

    @property
    def tokens_captured(self) -> int:
        return sum(r.shape[0] for r in self._routing)

    def finish(self) -> RouterTrace:
        """assemble everything captured into a trace."""
        if not self._routing:
            raise RuntimeError("no routing was captured, so no forward pass reached a router")

        routing = np.concatenate(self._routing)[: self.config.max_tokens]
        hidden = None
        if self._hidden:
            hidden = np.concatenate(self._hidden)[: self.config.max_tokens]

        # a boundary past the truncation point describes text that is not in the
        # trace, so drop it rather than carrying an index nothing can use
        segments = [s for s in self._segments if s < routing.shape[0]]
        segments_arr = np.array(sorted(set(segments)), dtype=np.int64) if segments else None

        return RouterTrace(
            routing=routing,
            n_experts=self.n_experts,
            hidden=hidden,
            segments=segments_arr,
            provenance=Provenance(
                source=SOURCE_CAPTURED,
                model_id=self.config.model_id,
                corpus=self.config.corpus,
                notes=(
                    f"{len(self.routers)} routers hooked, top-{self.top_k} of "
                    f"{self.n_experts}, hidden projected to {self.config.probe_dim}"
                ),
            ),
        )


# --------------------------------------------------------------- discovery


#: attribute names a decoder block gives its router, across families
ROUTER_LEAVES = {"router", "gate", "gating", "switch", "gate_proj_router"}

#: substrings that identify a router by its class name
ROUTER_CLASS_HINTS = ("router", "gating", "topkgate", "moegate")


def _can_emit(module: Any, n_experts: int) -> bool:
    """whether this module could plausibly produce `n_experts` scores.

    the verification step. without it, anything called `gate` gets hooked,
    including the gate projection of an ordinary swiglu mlp, which has nothing
    to do with routing and would fill the trace with garbage.
    """
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return module.out_features == n_experts

    weight = getattr(module, "weight", None)
    if weight is not None and getattr(weight, "ndim", 0) >= 1 and weight.shape[0] == n_experts:
        return True

    return any(
        isinstance(child, nn.Linear) and child.out_features == n_experts
        for child in module.modules()
    )


def find_routers(model: Any, n_experts: int | None = None) -> list[tuple[str, Any]]:
    """locate every router module.

    a candidate has to look like a router *and* be able to emit one score per
    expert. the name test alone is not enough, because a swiglu mlp calls one of
    its projections a gate, and the width test alone is not enough either, since
    plenty of matrices happen to have that many rows.

    raises rather than guessing whenever the result is ambiguous. a wrong hook
    point produces a complete and entirely fictional trace, so failing loudly is
    much cheaper than the alternative.
    """
    if n_experts is None:
        n_experts = infer_n_experts(model)

    candidates: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1].lower()
        cls = type(module).__name__.lower()
        looks_right = leaf in ROUTER_LEAVES or any(h in cls for h in ROUTER_CLASS_HINTS)
        if not looks_right:
            continue
        if "proj" in leaf and leaf not in ROUTER_LEAVES:
            continue
        if _can_emit(module, n_experts):
            candidates.append((name, module))

    # when a router module contains the linear that does the work, both match.
    # keep the inner one, whose output is the logits and nothing else.
    names = {n for n, _ in candidates}
    candidates = [
        (n, m)
        for n, m in candidates
        if not any(other != n and other.startswith(n + ".") for other in names)
    ]

    if not candidates:
        raise RuntimeError(
            "no router modules found. this model may not be a mixture of experts, "
            "or it may name its routers in a way this heuristic does not cover. "
            "pass the modules explicitly rather than letting it guess, because a "
            "wrong hook point produces a complete trace of fictional numbers."
        )
    return candidates


def infer_n_experts(model: Any) -> int:
    """expert count, from the config.

    read from the config rather than from a weight shape, because the shape is
    only meaningful once the right module has been found and finding the right
    module needs this number.
    """
    for attr in (
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
        "num_experts_per_layer",
    ):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError(
        "could not read the expert count from the model config. pass it "
        "explicitly rather than letting it guess."
    )


def infer_top_k(model: Any) -> int:
    """experts routed per token, from the config."""
    for attr in (
        "num_experts_per_tok",
        "moe_top_k",
        "top_k",
        "num_selected_experts",
        "n_experts_per_token",
    ):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError(
        "could not read top-k from the model config. pass it explicitly in "
        "CaptureConfig(top_k=...) rather than letting it guess, because a wrong "
        "top-k silently changes every number m0 reports."
    )


def capture_from_texts(
    config: CaptureConfig,
    texts: list[str],
    segment_starts: set[int] | None = None,
) -> RouterTrace:
    """load the model, run the texts through it, and return the trace.

    the convenience path. anything more involved than a list of strings is
    better served by driving [`RouterCapture`] directly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    kwargs: dict[str, Any] = dict(config.extra_model_kwargs)
    dispatched = config.max_memory is not None or "device_map" in kwargs
    if config.max_memory is not None:
        kwargs.setdefault("device_map", "auto")
        kwargs.setdefault("max_memory", {config.device: config.max_memory})
        kwargs.setdefault("offload_folder", config.offload_dir)
        kwargs.setdefault("low_cpu_mem_usage", True)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=getattr(torch, config.dtype),
        **kwargs,
    )
    model.eval()

    # a dispatched model is already placed, and accelerate raises if it is moved
    # again. only the all-in-memory path owns its placement.
    if not dispatched:
        model.to(config.device)

    capture = RouterCapture(model, config)
    starts = segment_starts or set()
    with capture, torch.no_grad():
        for i, text in enumerate(texts):
            if capture.tokens_captured >= config.max_tokens:
                break
            if i in starts:
                capture.mark_segment()
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            model(**{k: v.to(config.device) for k, v in batch.items()})
    return capture.finish()
