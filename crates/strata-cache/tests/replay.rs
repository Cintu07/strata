//! replay a real router trace through the shipping policy.
//!
//! every test here is `#[ignore]` and `scripts/test.sh` runs them in release:
//!
//! ```text
//! cargo test -p strata-cache --test replay --release -- --ignored --nocapture
//! ```
//!
//! they are ignored by default because each one replays 413,568 accesses
//! through five policies including an offline optimum, which takes seconds in
//! release and minutes in debug. a default suite people stop running is worse
//! than one that covers slightly less, so the coverage moved to a release step
//! rather than being deleted.
//!
//! # why this exists
//!
//! `tests/common/mod.rs` opens by admitting what it is: "synthetic access
//! traces with the structure real router traces are *claimed* to have". that
//! was honest when there were no real traces. there is one now, and it turned
//! out to carry a property none of the synthetic workloads were built with.
//!
//! a decoder walks every layer of every token in order, so one token is a scan
//! over `n_layers * top_k` distinct expert-layer pairs. on the captured granite
//! trace that is 192 pairs. any cache smaller than that sees a cyclic scan and
//! pure recency evicts every entry exactly before its next use, which takes lru
//! to zero rather than merely low. the synthetic generators produce hot sets and
//! domain blocks and skew, and none of them produce that.
//!
//! so this is the file that turns "claimed to" into "measured against".

mod common;

use common::uniform_sizes;
use std::path::PathBuf;
use strata_cache::baseline::{BeladyOracle, LfuCache, LruCache, Policy, without_admission};
use strata_cache::{CacheConfig, ExpertCache, ExpertDesc};
use strata_format::{ExpertKey, RouteTrace};

const EXPERT_BYTES: u64 = 1 << 20;

/// the target budget from g2: 20 percent of the model resident.
const TARGET_CAP: u64 = 154;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("granite.route")
}

/// read the trace the m0 harness exported, or say how to regenerate it.
fn load() -> RouteTrace {
    let path = fixture();
    RouteTrace::load(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\nregenerate it with:\n  \
             cd m0 && python -m strata_m0 export --trace traces/granite-full.npz",
            path.display()
        )
    })
}

struct Row {
    lru: f64,
    lfu: f64,
    no_admission: f64,
    strata: f64,
    oracle: f64,
    /// lru's raw hit count, because "exactly zero" is a claim about an integer
    /// and asserting it on a ratio is a float comparison pretending otherwise.
    lru_hits: u64,
}

fn run(trace: &[ExpertKey], capacity_experts: u64) -> Row {
    let capacity = capacity_experts * EXPERT_BYTES;
    let desc = ExpertDesc::plain(EXPERT_BYTES);

    let mut strata: ExpertCache<()> = ExpertCache::new(CacheConfig::with_capacity(capacity));
    let mut no_admission = without_admission(capacity);
    let mut lru = LruCache::new(capacity);
    let mut lfu = LfuCache::new(capacity);
    for &k in trace {
        strata.access(k, desc);
        no_admission.access(k, desc);
        lru.access(k, desc);
        lfu.access(k, desc);
    }
    let oracle = BeladyOracle::new(capacity).run(trace, &uniform_sizes(trace, EXPERT_BYTES));

    Row {
        lru: lru.stats().hit_rate(),
        lfu: lfu.stats().hit_rate(),
        no_admission: no_admission.stats().hit_rate(),
        strata: strata.stats().hit_rate(),
        oracle: oracle.hit_rate(),
        lru_hits: lru.stats().hits,
    }
}

#[test]
#[ignore = "replays 413k accesses through five policies; run in release, see module docs"]
fn real_trace_policy_table() {
    let trace = load();
    let total = trace.total_pairs() as u64;
    let working = trace.working_set();

    println!(
        "\ngranite-3.1-1b-a400m, {} tokens x {} layers x top-{} of {} experts",
        trace.n_tokens, trace.n_layers, trace.top_k, trace.n_experts
    );
    println!("{total} distinct expert-layer pairs, {working} touched per token\n");
    println!("  cap    % of model      lru      lfu   no-admission   strata   oracle");

    for frac in [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75] {
        let cap = ((total as f64) * frac).round() as u64;
        let r = run(trace.keys(), cap);
        println!(
            "  {cap:>3}  {:>9.0}%  {:>7.3}  {:>7.3}   {:>12.3}  {:>7.3}  {:>7.3}",
            frac * 100.0,
            r.lru,
            r.lfu,
            r.no_admission,
            r.strata,
            r.oracle
        );
    }

    println!("\nthe cliff, either side of one token's working set of {working}\n");
    for cap in [
        (working - 2) as u64,
        (working - 1) as u64,
        working as u64,
        (working + 1) as u64,
        (working + 8) as u64,
    ] {
        let r = run(trace.keys(), cap);
        println!(
            "  {cap:>3}             {:>7.3}  {:>7.3}   {:>12.3}  {:>7.3}  {:>7.3}",
            r.lru, r.lfu, r.no_admission, r.strata, r.oracle
        );
    }
    println!();
}

#[test]
#[ignore = "replays 413k accesses through five policies; run in release, see module docs"]
fn lru_returns_exactly_zero_below_one_token_of_working_set() {
    let trace = load();
    let r = run(trace.keys(), TARGET_CAP);

    // not "low". zero. every entry is evicted exactly before its next use.
    assert_eq!(
        r.lru_hits, 0,
        "lru should score exactly zero hits on a cyclic scan wider than the cache"
    );

    // and it is the scan, not the trace: give it one token's working set and
    // recency starts working again immediately.
    let above = run(trace.keys(), trace.working_set() as u64);
    assert!(
        above.lru > 0.3,
        "lru at the working set of {} should recover, got {:.3}",
        trace.working_set(),
        above.lru
    );
}

#[test]
#[ignore = "replays 413k accesses through five policies; run in release, see module docs"]
fn admission_control_is_what_survives_the_scan() {
    let trace = load();
    let r = run(trace.keys(), TARGET_CAP);

    // strata 0.365 against 0.212 with admission switched off, measured. the
    // eviction score alone does not survive the scan; the admission filter is
    // what keeps a one-shot expert from evicting a hot one.
    assert!(
        r.strata - r.no_admission > 0.13,
        "admission should be worth more than 0.13 here: strata {:.3}, \
         no-admission {:.3}",
        r.strata,
        r.no_admission
    );
    assert!(r.oracle > r.strata, "belady is the ceiling");
}

#[test]
#[ignore = "replays 413k accesses through five policies; run in release, see module docs"]
fn lfu_beats_strata_on_the_real_trace_and_that_is_the_open_problem() {
    let trace = load();

    // this asserts a result strata loses, on purpose, the same way the
    // synthetic suite asserts the workload where it trails lru. it is here so
    // that anyone changing the policy finds out whether they fixed this or
    // merely moved it.
    //
    // measured: lfu leads at every capacity up to 50 percent of the model and
    // the two converge only at 75 percent, where the cache is large enough that
    // policy barely matters.
    for (cap, min_gap) in [(38u64, 0.06), (77, 0.04), (TARGET_CAP, 0.03), (384, 0.002)] {
        let r = run(trace.keys(), cap);
        assert!(
            r.lfu - r.strata > min_gap,
            "at cap {cap} lfu {:.3} was expected to lead strata {:.3} by more \
             than {min_gap}. if this now fails because strata improved, that is \
             the good outcome: update the number and say so in the decision log.",
            r.lfu,
            r.strata
        );
    }

    // where strata does win, it wins narrowly and only once the cache is big.
    let big = run(trace.keys(), 576);
    assert!(
        big.strata >= big.lfu,
        "at 75 percent strata {:.3} should be at least lfu {:.3}",
        big.strata,
        big.lfu
    );
}

#[test]
#[ignore = "replays 413k accesses through five policies; run in release, see module docs"]
fn the_replay_agrees_with_the_python_simulator() {
    // the python harness measured the same trace with an independent
    // implementation. lru at 191 and 192 pairs, and belady at the target
    // budget, are the three numbers both sides report, and they are how we know
    // neither simulator is quietly wrong.
    let trace = load();

    let below = run(trace.keys(), 191);
    let at = run(trace.keys(), 192);
    assert!(
        (below.lru - 0.198).abs() < 0.005,
        "python measured lru 0.198 at 191 pairs, rust says {:.3}",
        below.lru
    );
    assert!(
        (at.lru - 0.327).abs() < 0.005,
        "python measured lru 0.327 at 192 pairs, rust says {:.3}",
        at.lru
    );

    let target = run(trace.keys(), TARGET_CAP);
    assert!(
        (target.oracle - 0.561).abs() < 0.01,
        "python measured belady 0.561 at the target budget, rust says {:.3}",
        target.oracle
    );
    assert!(
        (target.lfu - 0.398).abs() < 0.01,
        "python measured lfu 0.398 at the target budget, rust says {:.3}",
        target.lfu
    );
}
