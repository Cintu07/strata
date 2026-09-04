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
