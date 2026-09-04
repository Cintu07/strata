"""does the capture path actually work on a real moe implementation?

the analyses in `test_harness.py` are checked against synthetic traces. this
file checks the other half: that the router hooks find the right modules in a
real hugging face mixture-of-experts model and record what they claim to.

it does that without downloading anything, by building a genuinely real model
class from a shrunken config with random weights. the weights are meaningless,
so nothing here is a measurement. the module structure, the hook points, the
tensor shapes and the top-k discovery are all exactly what the full sized model
has, and those are the things that break.

skipped when torch and transformers are not installed, since they are the
`[capture]` extra and the analysis half does not need them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from strata_m0.capture import (  # noqa: E402
    CaptureConfig,
    RouterCapture,
    extract_routing,
    find_routers,
    infer_n_experts,
    infer_top_k,
)

N_LAYERS = 4
N_EXPERTS = 8
TOP_K = 2
HIDDEN = 32
SEQ = 24


def tiny_granite_moe():
    """a real GraniteMoeForCausalLM, small enough to build in a test."""
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

    config = GraniteMoeConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return GraniteMoeForCausalLM(config).eval()


@pytest.fixture(scope="module")
def model():
    try:
        return tiny_granite_moe()
    except Exception as e:  # pragma: no cover - depends on the installed version
        pytest.skip(f"could not build a GraniteMoe model here: {e}")


def test_routers_are_found_one_per_layer(model):
    routers = find_routers(model)
    assert len(routers) == N_LAYERS, f"found {[n for n, _ in routers]}"

    # granitemoe's router is not an nn.Linear, it holds a bare weight and calls
    # F.linear itself, which is exactly why discovery cannot key on the type.
    # what has to hold is that the module can emit one score per expert.
    for name, module in routers:
        width = getattr(module, "out_features", None)
        if width is None:
            weight = getattr(module, "weight", None)
            assert weight is not None, f"{name} has neither out_features nor a weight"
            width = weight.shape[0]
        assert width == N_EXPERTS, f"{name} has width {width}, expected {N_EXPERTS}"

    # and one per decoder block, not one per model and not one per expert
    assert len({n.rsplit(".", 1)[0] for n, _ in routers}) == N_LAYERS


def test_the_attention_projections_are_not_mistaken_for_routers(model):
    # a swiglu mlp calls one of its projections a gate, and the attention block
    # is full of bias free linears. neither is a router, and hooking one would
    # fill the trace with numbers that are not routing decisions at all.
    names = {n for n, _ in find_routers(model)}
    assert not any("self_attn" in n for n in names), names
    assert not any(n.endswith("gate_proj") for n in names), names


def test_expert_count_and_top_k_come_from_the_config(model):
    assert infer_n_experts(model) == N_EXPERTS
    assert infer_top_k(model) == TOP_K


class _Stub:
    """the smallest thing that looks like a model to the config readers."""

    def __init__(self, **kwargs):
        self.config = type("Cfg", (), kwargs)()


def test_top_k_failure_is_loud_rather_than_a_guess():
    # a wrong top-k silently changes every number m0 reports, so the code has to
    # refuse rather than pick something plausible
    with pytest.raises(RuntimeError, match="top-k"):
        infer_top_k(_Stub(num_local_experts=8))


def test_expert_count_failure_is_loud_too():
    with pytest.raises(RuntimeError, match="expert count"):
        infer_n_experts(_Stub(num_experts_per_tok=2))


def test_a_forward_pass_produces_a_well_formed_trace(model):
    config = CaptureConfig(
        model_id="tiny-granite-moe",
        corpus="random token ids",
        max_tokens=SEQ,
        probe_dim=16,
    )
    capture = RouterCapture(model, config)
    ids = torch.randint(0, 256, (1, SEQ))

    with capture, torch.no_grad():
        model(input_ids=ids)

    trace = capture.finish()

    assert trace.n_tokens == SEQ
    assert trace.n_layers == N_LAYERS
    assert trace.top_k == TOP_K
    assert trace.n_experts == N_EXPERTS
    assert trace.provenance.is_real, "a captured trace is a real one"

    assert trace.has_hidden
    assert trace.hidden.shape == (SEQ, N_LAYERS, 16)

    # every recorded expert index has to be routable, or the hook read the wrong
    # tensor and the whole trace is fiction
    assert trace.routing.min() >= 0
    assert trace.routing.max() < N_EXPERTS


def test_the_recorded_choices_match_the_routers_own_decision(model):
    """the hook must record what the router actually chose.

    this is the test that catches reading the wrong tensor out of a router's
    return value, which is the failure that produces a complete, well shaped and
    entirely fictional trace. it caught exactly that: granitemoe returns
    `(top_k_index, top_k_weights, router_logits)`, and taking element zero and
    running top-k over it treats expert indices as scores.

    truth here is taken from the router logits, identified by dtype and width
    rather than by position, and cross checked against the index tensor the
    router itself returned. the two agreeing is what makes this independent of
    the code under test.
    """
    config = CaptureConfig(model_id="tiny", max_tokens=SEQ, probe_dim=0)
    capture = RouterCapture(model, config)
    routers = find_routers(model)

    seen: dict[int, tuple] = {}

    def record(index):
        def hook(_m, _inputs, output):
            seen[index] = output

        return hook

    handles = [m.register_forward_hook(record(i)) for i, (_, m) in enumerate(routers)]
    ids = torch.randint(0, 256, (1, SEQ))
    with capture, torch.no_grad():
        model(input_ids=ids)
    for h in handles:
        h.remove()

    trace = capture.finish()

    for layer in range(N_LAYERS):
        out = seen[layer]
        tensors = list(out) if isinstance(out, tuple) else [out]
        tensors = [t for t in tensors if torch.is_tensor(t)]

        logits = next(t for t in tensors if t.is_floating_point() and t.shape[-1] == N_EXPERTS)
        from_logits = torch.topk(logits.reshape(-1, N_EXPERTS), TOP_K, dim=1).indices

        chosen = [t for t in tensors if not t.is_floating_point() and t.shape[-1] == TOP_K]
        if chosen:
            own = chosen[0].reshape(-1, TOP_K)
            for token in range(SEQ):
                assert set(own[token].tolist()) == set(from_logits[token].tolist()), (
                    "the router's own indices disagree with its own logits, so the "
                    "reference for this test is not trustworthy"
                )

        for token in range(SEQ):
            expected = set(from_logits[token].tolist())
            got = set(trace.routing[token, layer].tolist())
            assert got == expected, (
                f"layer {layer} token {token}: captured {sorted(got)} "
                f"but the router chose {sorted(expected)}"
            )


def test_routing_is_read_by_dtype_and_width_not_by_position():
    """the extraction must not depend on where a family puts its tensors."""
    logits = torch.randn(5, 8)
    indices = torch.tensor([[1, 2]] * 5, dtype=torch.int64)
    weights = torch.rand(5, 2)

    # a bare logits tensor, as mixtral's gate returns
    kind, t = extract_routing(logits, n_experts=8, top_k=2)
    assert kind == "logits" and t.shape == (5, 8)

    # granitemoe's three tuple, where element zero is the indices
    kind, t = extract_routing((indices, weights, logits), n_experts=8, top_k=2)
    assert kind == "chosen", "the router's own decision should be preferred"
    assert torch.equal(t, indices)

    # and something that is neither must fail rather than return a guess
    with pytest.raises(RuntimeError, match="routing decision"):
        extract_routing(torch.randn(5, 3), n_experts=8, top_k=2)


def test_capture_without_hidden_states_stays_small(model):
    config = CaptureConfig(model_id="tiny", max_tokens=SEQ, probe_dim=0)
    capture = RouterCapture(model, config)
    with capture, torch.no_grad():
        model(input_ids=torch.randint(0, 256, (1, SEQ)))
    trace = capture.finish()
    assert not trace.has_hidden
    assert trace.routing.shape == (SEQ, N_LAYERS, TOP_K)


def test_capturing_nothing_fails_loudly(model):
    capture = RouterCapture(model, CaptureConfig(model_id="tiny", probe_dim=0))
    with pytest.raises(RuntimeError, match="no routing was captured"):
        capture.finish()
