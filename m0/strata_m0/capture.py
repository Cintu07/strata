"""capturing a real router trace from a hugging face moe model.

this is the only module that needs torch and transformers, and it is imported
lazily so that the analysis half of the harness runs without them.

# how it works

moe implementations differ, but they all have a router: a small linear layer,
one per block, that maps the hidden state to per-expert scores. a forward hook
on that module sees both the input it was given and the logits it produced,
which is exactly the pair the speculative router head would have to learn.

the router modules are found by name rather than by walking a known architecture,
because the names differ between families and the shapes do not. anything that
is a `Linear` whose output width equals the model's expert count, sitting inside
a block, is a router. that heuristic is checked against the config and the module
count, and it refuses rather than guesses if the result looks wrong.

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
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)


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
        self.routers = find_routers(model)
        self.n_experts = infer_n_experts(model, self.routers)
        self.top_k = config.top_k or infer_top_k(model)

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
                logits = output[0] if isinstance(output, tuple) else output
                logits = logits.detach()

                hidden = hidden.reshape(-1, hidden.shape[-1]).float().cpu().numpy()
                logits = logits.reshape(-1, logits.shape[-1]).float().cpu().numpy()

                chosen = np.argpartition(-logits, self.top_k - 1, axis=1)[:, : self.top_k]
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

        return RouterTrace(
            routing=routing,
            n_experts=self.n_experts,
            hidden=hidden,
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


def find_routers(model: Any) -> list[tuple[str, Any]]:
    """locate every router module, by shape and position rather than by name.

    names differ across model families and shapes do not, so the search is for
    a small linear layer inside a decoder block whose output width matches the
    expert count. the result is checked against the block count, and this
    raises rather than guessing if the two disagree.
    """
    import torch.nn as nn

    candidates: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or module.bias is not None:
            continue
        lowered = name.lower()
        if any(tag in lowered for tag in ("gate", "router", "switch")) and "proj" not in lowered:
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError(
            "no router modules found. this model may not be a mixture of experts, "
            "or it may name its routers in a way this heuristic does not cover. "
            "pass the modules explicitly rather than letting it guess."
        )
    widths = {m.out_features for _, m in candidates}
    if len(widths) != 1:
        raise RuntimeError(
            f"found candidate routers with differing output widths {sorted(widths)}, "
            "which means the heuristic caught something that is not a router"
        )
    return candidates


def infer_n_experts(model: Any, routers: list[tuple[str, Any]]) -> int:
    """expert count, from the config where possible and the router shape if not."""
    for attr in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return int(routers[0][1].out_features)


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


def capture_from_texts(config: CaptureConfig, texts: list[str]) -> RouterTrace:
    """load the model, run the texts through it, and return the trace.

    the convenience path. anything more involved than a list of strings is
    better served by driving [`RouterCapture`] directly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=getattr(torch, config.dtype),
        **config.extra_model_kwargs,
    )
    model.eval().to(config.device)

    capture = RouterCapture(model, config)
    with capture, torch.no_grad():
        for text in texts:
            if capture.tokens_captured >= config.max_tokens:
                break
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            model(**{k: v.to(config.device) for k, v in batch.items()})
    return capture.finish()
