"""does the harness recover structure that was deliberately put there?

every test here plants a known answer in a synthetic trace and checks that the
analysis finds it. an analysis that cannot recover a structure someone put in on
purpose will not find one that a real model put in by accident, and it is much
cheaper to discover that here than after three weeks of capture.

the negative cases matter at least as much. an analysis that reports structure
in a trace built to have none is not a measurement, it is a leak.
"""

from __future__ import annotations

import numpy as np
import pytest

from strata_m0 import RouterTrace, analysis, cache_sim, predict, report
from strata_m0.synthetic import make_trace
from strata_m0.trace import SOURCE_SYNTHETIC, Provenance


# ------------------------------------------------------------------- trace


def test_trace_round_trips_through_disk(tmp_path):
    trace = make_trace(n_tokens=200, n_layers=4, n_experts=16, top_k=2, seed=1)
    path = trace.save(tmp_path / "t.npz")
    loaded = RouterTrace.load(path)

    assert np.array_equal(loaded.routing, trace.routing)
    assert loaded.n_experts == trace.n_experts
    assert loaded.provenance.source == SOURCE_SYNTHETIC
    np.testing.assert_allclose(loaded.hidden, trace.hidden)


def test_a_trace_without_hidden_states_round_trips_too(tmp_path):
    trace = make_trace(n_tokens=100, n_layers=3, n_experts=8, top_k=2, with_hidden=False)
    loaded = RouterTrace.load(trace.save(tmp_path / "t.npz"))
    assert loaded.hidden is None
    assert not loaded.has_hidden


def test_an_out_of_range_expert_is_rejected_rather_than_silently_wrapped():
    routing = np.array([[[0, 99]]], dtype=np.int32)
    with pytest.raises(ValueError, match="expert 99"):
        RouterTrace(routing, n_experts=8, provenance=Provenance(SOURCE_SYNTHETIC, "x"))


def test_packed_keys_separate_layers():
    trace = make_trace(n_tokens=10, n_layers=3, n_experts=8, top_k=2, with_hidden=False)
    keys = trace.flat_keys()
    assert keys.size == trace.n_tokens * trace.n_layers * trace.top_k

    # layer 0 expert 5 and layer 2 expert 5 must not collide, which is the whole
    # reason the key is a pair and not an index
    assert ((0 << 32) | 5) != ((2 << 32) | 5)
    layers = keys >> 32
    assert set(layers.tolist()) == {0, 1, 2}


def test_provenance_marks_synthetic_traces_unmistakably():
    trace = make_trace(n_tokens=50, with_hidden=False)
    assert not trace.provenance.is_real
    assert "SYNTHETIC" in trace.provenance.banner()


# ------------------------------------------------------------------- reuse


def test_reuse_rises_with_the_persistence_that_was_planted():
    low = make_trace(n_tokens=800, persistence=0.0, seed=2, with_hidden=False)
    high = make_trace(n_tokens=800, persistence=0.9, seed=2, with_hidden=False)

    assert analysis.reuse_across_tokens(high).overall > (
        analysis.reuse_across_tokens(low).overall + 0.2
    )


def test_reuse_of_a_constant_trace_is_total():
    routing = np.tile(np.array([[[0, 1, 2, 3]]], dtype=np.int32), (100, 6, 1))
    trace = RouterTrace(routing, 32, Provenance(SOURCE_SYNTHETIC, "constant"))
    assert analysis.reuse_across_tokens(trace).overall == pytest.approx(1.0)


def test_reuse_of_disjoint_alternating_sets_is_zero():
    a = np.array([0, 1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6, 7], dtype=np.int32)
    routing = np.stack([a if t % 2 == 0 else b for t in range(100)])[:, None, :]
    trace = RouterTrace(routing, 32, Provenance(SOURCE_SYNTHETIC, "alternating"))
    assert analysis.reuse_across_tokens(trace).overall == pytest.approx(0.0)


def test_the_reuse_curve_falls_off_with_distance():
    trace = make_trace(n_tokens=1200, persistence=0.6, domain_block=100, seed=4,
                       with_hidden=False)
    curve = analysis.reuse_curve(trace, max_distance=12)
    assert curve[0] > curve[-1], "the nearest token should be the most informative"


# -------------------------------------------------------------------- skew


def test_skew_is_detected_when_planted_and_absent_when_not():
    flat = make_trace(n_tokens=800, skew=0.0, persistence=0.0, seed=6, with_hidden=False)
    peaked = make_trace(n_tokens=800, skew=4.0, persistence=0.0, seed=6, with_hidden=False)

    flat_gini = analysis.access_skew(flat).gini.mean()
    peaked_gini = analysis.access_skew(peaked).gini.mean()
    assert peaked_gini > flat_gini + 0.1


def test_a_perfectly_uniform_router_has_near_zero_gini():
    n_experts = 16
    routing = np.tile(np.arange(n_experts, dtype=np.int32), (200, 1))[:, None, :]
    trace = RouterTrace(routing, n_experts, Provenance(SOURCE_SYNTHETIC, "uniform"))
    skew = analysis.access_skew(trace)
    assert skew.gini[0] == pytest.approx(0.0, abs=1e-9)
    assert skew.normalised_entropy[0] == pytest.approx(1.0, abs=1e-9)


def test_a_degenerate_router_is_maximally_skewed():
    routing = np.zeros((200, 1, 1), dtype=np.int32)
    trace = RouterTrace(routing, 64, Provenance(SOURCE_SYNTHETIC, "degenerate"))
    skew = analysis.access_skew(trace)
    assert skew.normalised_entropy[0] == pytest.approx(0.0)
    assert skew.top_decile_mass[0] == pytest.approx(1.0)


# ------------------------------------------------------------ co-activation


def test_co_activation_lift_finds_groups_that_always_fire_together():
    # two fixed groups, chosen at random, so experts inside a group are perfectly
    # correlated and experts across groups never co-occur
    rng = np.random.default_rng(0)
    groups = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]
    routing = np.stack([groups[rng.integers(2)] for _ in range(600)])[:, None, :].astype(np.int32)
    trace = RouterTrace(routing, 32, Provenance(SOURCE_SYNTHETIC, "grouped"))

    result = analysis.coactivation(trace, 0)
    assert result.lift > 2.0, f"structure this strong should show clear lift, got {result.lift}"


def test_independent_routing_shows_no_meaningful_lift():
    rng = np.random.default_rng(1)
    routing = rng.integers(0, 32, size=(3000, 1, 4)).astype(np.int32)
    trace = RouterTrace(routing, 32, Provenance(SOURCE_SYNTHETIC, "independent"))

    result = analysis.coactivation(trace, 0)
    assert result.lift < 1.4, f"there is nothing here to find, but lift was {result.lift}"


# --------------------------------------------------------------- cache sim


def test_belady_is_never_beaten_by_an_online_policy():
    trace = make_trace(n_tokens=600, n_layers=6, n_experts=24, top_k=3, seed=8,
                       with_hidden=False)
    keys = trace.flat_keys()
    for capacity in (8, 32, 64):
        optimal = cache_sim.belady(keys, capacity).hit_rate
        assert cache_sim.lru(keys, capacity).hit_rate <= optimal + 1e-12
        assert cache_sim.lfu(keys, capacity).hit_rate <= optimal + 1e-12


def test_a_cache_holding_everything_misses_only_on_first_touch():
    trace = make_trace(n_tokens=300, n_layers=4, n_experts=16, top_k=2, seed=9,
                       with_hidden=False)
    keys = trace.flat_keys()
    distinct = np.unique(keys).size
    result = cache_sim.lru(keys, distinct)
    assert result.misses == distinct
    assert result.hits == keys.size - distinct


def test_a_cache_of_zero_hits_nothing():
    keys = np.arange(100, dtype=np.int64)
    for policy in (cache_sim.lru, cache_sim.lfu, cache_sim.belady):
        assert policy(keys, 0).hit_rate == 0.0


def test_hit_rate_rises_monotonically_with_cache_size():
    trace = make_trace(n_tokens=500, n_layers=5, n_experts=20, top_k=3, seed=10,
                       with_hidden=False)
    results = cache_sim.sweep(trace, capacities=[4, 8, 16, 32, 64, 100])
    for series in results.values():
        rates = [r.hit_rate for r in series]
        assert rates == sorted(rates), f"{series[0].policy} was not monotonic: {rates}"


def test_the_knee_reports_the_smallest_cache_reaching_the_target():
    trace = make_trace(n_tokens=500, n_layers=4, n_experts=16, top_k=2, seed=11,
                       with_hidden=False)
    series = cache_sim.sweep(trace, capacities=[2, 4, 8, 16, 32, 64])["belady"]
    knee = cache_sim.knee(series, target=0.7)
    if knee is not None:
        smaller = [r for r in series if r.capacity < knee]
        assert all(r.hit_rate < 0.7 for r in smaller)


# --------------------------------------------------------------- prediction


def test_a_probe_recovers_signal_that_was_planted():
    trace = make_trace(
        n_tokens=1500, n_layers=8, n_experts=32, top_k=4,
        persistence=0.0, signal=1.0, seed=12,
    )
    results = predict.lookahead_recall(trace, k_values=(4,))
    assert results, "k=4 should be measurable on an 8 layer trace"
    assert results[0].best_probe() > 0.6, (
        f"with a perfectly informative hidden state the probe should do well, "
        f"got {results[0].best_probe():.3f}"
    )


def test_a_probe_finds_nothing_in_a_hidden_state_that_carries_nothing():
    # signal 0 means the recorded hidden state is pure noise, so anything much
    # above the chance rate would be a leak in the evaluation rather than a result
    trace = make_trace(
        n_tokens=1200, n_layers=6, n_experts=32, top_k=4,
        persistence=0.0, signal=0.0, seed=13,
    )
    results = predict.lookahead_recall(trace, k_values=(2,))
    chance = results[0].budget / trace.n_experts
    assert results[0].best_probe() < chance + 0.25, (
        f"noise should not be predictable: got {results[0].best_probe():.3f} "
        f"against a chance rate of {chance:.3f}"
    )


def test_the_persistence_prior_is_measured_and_tracks_what_was_planted():
    weak = predict.lookahead_recall(
        make_trace(n_tokens=800, n_layers=6, persistence=0.0, seed=14), k_values=(1,)
    )
    strong = predict.lookahead_recall(
        make_trace(n_tokens=800, n_layers=6, persistence=0.9, seed=14), k_values=(1,)
    )
    assert strong[0].persistence > weak[0].persistence


def test_prediction_without_hidden_states_fails_loudly():
    trace = make_trace(n_tokens=100, with_hidden=False)
    with pytest.raises(ValueError, match="no hidden states"):
        predict.lookahead_recall(trace)


def test_recall_at_a_full_budget_is_one():
    scores = np.random.default_rng(0).normal(size=(50, 16))
    truth = np.zeros((50, 16), dtype=np.float32)
    truth[:, :3] = 1.0
    assert predict.recall_at(scores, truth, budget=16) == pytest.approx(1.0)


# ------------------------------------------------------------------ report


def test_the_report_runs_end_to_end_and_refuses_to_call_synthetic_a_result(tmp_path):
    trace = make_trace(n_tokens=600, n_layers=6, n_experts=24, top_k=3, seed=15)
    path = report.run(trace, tmp_path)

    text = path.read_text(encoding="utf-8")
    assert "this is not a measurement" in text
    assert "SYNTHETIC" in text

    for figure in ("reuse", "skew", "cache_curve", "coactivation", "prediction"):
        assert (tmp_path / "figures" / f"{figure}.png").exists(), f"{figure} was not drawn"
    assert (tmp_path / "summary.json").exists()


def test_the_summary_records_that_a_synthetic_run_is_not_real(tmp_path):
    import json

    trace = make_trace(n_tokens=400, n_layers=4, n_experts=16, top_k=2, seed=16)
    report.run(trace, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["is_real_measurement"] is False
    assert summary["checks"], "the verdict table should not be empty"
    assert all(c["verdict"] in {"pass", "marginal", "fail"} for c in summary["checks"])


# --------------------------------------------------------- domain structure


def test_domain_correlation_finds_structure_that_was_planted():
    # four domains in blocks, which is what the synthetic generator builds
    trace = make_trace(
        n_tokens=1200, n_layers=4, n_experts=32, top_k=4,
        n_domains=4, domain_block=300, persistence=0.0, seed=31, with_hidden=False,
    )
    result = analysis.domain_correlation(trace, boundaries=[0, 300, 600, 900], window=25)

    assert result.within > result.across, str(result)
    assert result.separation > 0.02, (
        f"the generator puts a distinct hot set in each domain, so this should "
        f"be clearly positive: {result}"
    )


def test_domain_correlation_finds_nothing_when_there_are_no_domains():
    # one domain, so any apparent separation is an artefact of the estimator
    trace = make_trace(
        n_tokens=1200, n_layers=4, n_experts=32, top_k=4,
        n_domains=1, domain_block=10_000, persistence=0.0, seed=32, with_hidden=False,
    )
    # the boundaries are a lie here on purpose: the text never changed subject
    result = analysis.domain_correlation(trace, boundaries=[0, 300, 600, 900], window=25)

    assert abs(result.separation) < 0.05, (
        f"there is no domain structure to find, so separation should be near "
        f"zero rather than positive: {result}"
    )


def test_domain_correlation_needs_at_least_two_domains():
    trace = make_trace(n_tokens=200, n_layers=2, n_experts=8, top_k=2, with_hidden=False)
    assert analysis.domain_correlation(trace, boundaries=[0]).separation == 0.0


def test_the_domain_metric_does_not_saturate_at_real_model_shape():
    """regression for the bug that made the first granite run report a false fail.

    the first version took ``np.unique`` over a window of the routing tensor,
    which both flattened the layers together and reduced to a set. at granite's
    shape a 64 token window draws twelve thousand times from 768 expert-layer
    pairs, so every pair appears, the jaccard index between any two windows is
    exactly 1.0, and the metric reported 1.000 within and 1.000 across.
    """
    trace = make_trace(
        n_tokens=1024, n_layers=24, n_experts=32, top_k=8,
        n_domains=4, domain_block=256, persistence=0.0, seed=33, with_hidden=False,
    )
    window = 64

    # the condition that broke the old metric is genuinely present here
    assert len(np.unique(trace.routing[:window])) == trace.n_experts

    result = analysis.domain_correlation(
        trace, boundaries=[0, 256, 512, 768], window=window
    )
    assert result.within < 0.999, f"the metric saturated again: {result}"
    assert result.separation > 0.0, str(result)


def test_the_null_rejects_separation_that_is_only_temporal_locality():
    """a corpus that never changes subject must not clear the null.

    windows inside a block are adjacent in time, so persistence alone produces a
    positive raw separation. that is the confound the circular shift exists to
    strip, and this is the test that says it does.
    """
    trace = make_trace(
        n_tokens=1600, n_layers=4, n_experts=32, top_k=4,
        n_domains=1, domain_block=10_000, persistence=0.7, seed=34, with_hidden=False,
    )
    # the boundaries are a lie on purpose: the text never changed subject
    null = analysis.domain_null(
        trace, boundaries=[0, 400, 800, 1200], window=25, n_shifts=200
    )

    assert null.p_value > 0.05, (
        f"there is no subject structure, only locality, so the shifted labels "
        f"should explain the observed separation: {null}"
    )


def test_the_null_confirms_separation_that_was_planted():
    trace = make_trace(
        n_tokens=1600, n_layers=4, n_experts=32, top_k=4,
        n_domains=4, domain_block=400, persistence=0.0, seed=35, with_hidden=False,
    )
    null = analysis.domain_null(
        trace, boundaries=[0, 400, 800, 1200], window=25, n_shifts=200
    )

    assert null.p_value <= 0.01, str(null)
    assert null.margin > 0.0, str(null)


def test_the_replay_export_round_trips_the_way_rust_will_read_it(tmp_path):
    """parse the export back the way `tests/replay.rs` does.

    the rust side rebuilds an ExpertKey from a layer and an expert index rather
    than reading a packed key, so the two sides cannot silently disagree about
    the packing. this checks the byte layout that agreement rests on.
    """
    trace = make_trace(
        n_tokens=40, n_layers=3, n_experts=8, top_k=2, seed=41, with_hidden=False
    )
    path = trace.write_replay(tmp_path / "t.route")
    raw = path.read_bytes()

    assert raw[0:8] == b"STRTRACE"
    fields = np.frombuffer(raw[8:28], dtype="<u4")
    version, n_tokens, n_layers, n_experts, top_k = fields.tolist()
    assert version == 1
    assert (n_tokens, n_layers, n_experts, top_k) == (40, 3, 8, 2)

    body = np.frombuffer(raw[28:], dtype="<u2")
    assert body.size == n_tokens * n_layers * top_k
    np.testing.assert_array_equal(
        body.reshape(n_tokens, n_layers, top_k).astype(np.int64), trace.routing
    )


def test_the_replay_export_refuses_more_experts_than_the_format_holds():
    trace = make_trace(n_tokens=10, n_layers=2, n_experts=8, top_k=2, with_hidden=False)
    trace.n_experts = 70_000
    with pytest.raises(ValueError, match="u16"):
        trace.write_replay("unused.route")


def test_the_static_prior_is_the_bar_when_routing_ignores_the_token():
    """a probe must be scored against the better of the two free baselines.

    granite showed why. the persistence prior read 0.623 at k=4 and the static
    frequency table read 0.718 on the same rows, so a probe at 0.752 looked like
    it cleared the bar by 0.129 when the honest margin was 0.034. scoring
    against the weaker baseline is how a project talks itself into shipping a
    model that buys nothing.
    """
    # skew makes some experts far more popular, and no persistence at all, so a
    # frequency table is the thing to beat and recency is not
    trace = make_trace(
        n_tokens=1200, n_layers=8, n_experts=32, top_k=4,
        persistence=0.0, skew=3.0, signal=0.0, seed=36,
    )
    k4 = next(r for r in predict.lookahead_recall(trace) if r.k == 4)

    assert k4.static > k4.persistence, (
        f"a skewed router with no persistence should favour the frequency "
        f"table: static {k4.static:.3f}, persistence {k4.persistence:.3f}"
    )
    assert k4.free_baseline() == max(k4.persistence, k4.static)
    assert k4.margin() == k4.best_probe() - k4.free_baseline()


def test_a_hidden_state_carrying_nothing_cannot_beat_the_free_baselines():
    trace = make_trace(
        n_tokens=1200, n_layers=8, n_experts=32, top_k=4,
        persistence=0.0, skew=3.0, signal=0.0, seed=37,
    )
    k4 = next(r for r in predict.lookahead_recall(trace) if r.k == 4)

    assert k4.margin() < 0.05, (
        f"the hidden state carries no routing signal, so a probe should not "
        f"clear the free baseline by anything worth shipping: {k4}"
    )


def test_the_null_is_empty_without_domains():
    trace = make_trace(n_tokens=200, n_layers=2, n_experts=8, top_k=2, with_hidden=False)
    assert analysis.domain_null(trace, boundaries=[0]).n_shifts == 0
