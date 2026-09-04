//! what the cache policy is claimed to do, stated as assertions.
//!
//! the thresholds here were read off `tests/measure.rs` rather than guessed.
//! they sit well clear of the measured values so that ordinary drift does not
//! trip them, and close enough that losing the mechanism would.

mod common;

use common::{domain_blocks, hot_set_with_noise, skewed_layer, uniform_sizes};
use std::collections::HashMap;
use strata_cache::baseline::{BeladyOracle, LruCache, Policy, without_admission};
use strata_cache::{CacheConfig, ExpertCache, ExpertDesc, Region, Residency};
use strata_format::ExpertKey;

const MB: u64 = 1 << 20;

fn strata(capacity: u64) -> ExpertCache<()> {
    ExpertCache::new(CacheConfig::with_capacity(capacity))
}

fn run<P: Policy>(policy: &mut P, trace: &[ExpertKey], bytes: u64) -> f64 {
    let desc = ExpertDesc::plain(bytes);
    for &k in trace {
        policy.access(k, desc);
    }
    policy.stats().hit_rate()
}

// ---------------------------------------------------------------- the claims

#[test]
fn admission_control_survives_one_shot_traffic_that_lru_does_not() {
    // a hot set of 16 revisited, with four experts per round that are touched
    // once and never again. this is the long tail of a prefill in miniature.
    let trace = hot_set_with_noise(16, 200, 4, 7);
    let capacity = 16 * MB;

    let lru = run(&mut LruCache::new(capacity), &trace, MB);
    let strata = run(&mut strata(capacity), &trace, MB);

    assert!(
        strata > lru + 0.10,
        "strata {strata:.3} should clear lru {lru:.3} by a wide margin on scan traffic"
    );
}

#[test]
fn frequency_beats_recency_on_skewed_access() {
    // load imbalance is the one thing every router paper agrees on
    for cap in [16u64, 24, 32] {
        let trace = skewed_layer(64, 20_000, 13);
        let capacity = cap * MB;

        let lru = run(&mut LruCache::new(capacity), &trace, MB);
        let strata = run(&mut strata(capacity), &trace, MB);

        assert!(
            strata > lru + 0.04,
            "at {cap} experts, strata {strata:.3} should beat lru {lru:.3}"
        );
    }
}

#[test]
fn admission_is_what_does_the_work_not_just_the_eviction_score() {
    // the ablation. if this stopped holding, the win would be coming from
    // somewhere other than where the design says it is.
    let trace = hot_set_with_noise(16, 200, 4, 7);
    let capacity = 16 * MB;

    let ablated = run(&mut without_admission(capacity), &trace, MB);
    let full = run(&mut strata(capacity), &trace, MB);

    assert!(
        full > ablated + 0.05,
        "admission should be worth real points: full {full:.3} vs ablated {ablated:.3}"
    );
}

#[test]
fn nothing_beats_the_offline_optimum() {
    let trace = skewed_layer(64, 20_000, 13);
    let sizes: HashMap<_, _> = uniform_sizes(&trace, MB);

    for cap in [8u64, 16, 32] {
        let capacity = cap * MB;
        let oracle = BeladyOracle::new(capacity).run(&trace, &sizes).hit_rate();
        let lru = run(&mut LruCache::new(capacity), &trace, MB);
        let strata = run(&mut strata(capacity), &trace, MB);

        // sizes are uniform here, which is the case where belady is exactly
        // optimal rather than an estimate
        assert!(
            strata <= oracle + 1e-9,
            "strata {strata:.3} exceeded the optimum {oracle:.3}"
        );
        assert!(
            lru <= oracle + 1e-9,
            "lru {lru:.3} exceeded the optimum {oracle:.3}"
        );
        assert!(oracle > 0.0);
    }
}

/// the honest one.
///
/// on a workload that is pure short range recency, with a topic running long
/// enough to fully turn the cache over and no reuse across topics, lru is very
/// hard to beat and strata does not beat it. admission is a cost here: every
/// new topic has to fight its way in past the previous one.
///
/// this is written down as a test rather than left out of the readme, because a
/// policy comparison that only reports the workloads it wins is not a
/// measurement. the ablation shows the cost is admission specifically, and the
/// knob to turn is in the config.
#[test]
fn a_pure_recency_workload_is_where_lru_still_wins() {
    let trace = domain_blocks(4, 12, 300, 24, 11);
    let capacity = 24 * MB;

    let lru = run(&mut LruCache::new(capacity), &trace, MB);
    let strata = run(&mut strata(capacity), &trace, MB);
    let ablated = run(&mut without_admission(capacity), &trace, MB);

    assert!(
        lru > strata,
        "lru {lru:.3} is expected to lead here, strata was {strata:.3}"
    );
    assert!(
        strata > 0.80,
        "but the gap must stay a trade, not a collapse: {strata:.3}"
    );
    assert!(
        ablated > lru - 0.05,
        "turning admission off should recover most of it: {ablated:.3} against {lru:.3}"
    );
}

// ------------------------------------------------------------- the invariants

#[test]
fn the_byte_ceiling_is_never_exceeded() {
    let trace = skewed_layer(64, 5_000, 3);
    let capacity = 12 * MB;
    let mut cache = strata(capacity);
    let desc = ExpertDesc::plain(MB);

    for &k in &trace {
        cache.access(k, desc);
        assert!(
            cache.resident_bytes() <= capacity,
            "held {} against a ceiling of {capacity}",
            cache.resident_bytes()
        );
    }
    assert!(!cache.is_empty());
}

#[test]
fn resident_bytes_matches_what_is_actually_held() {
    let trace = skewed_layer(32, 2_000, 5);
    let mut cache = strata(8 * MB);
    let desc = ExpertDesc::plain(MB);
    for &k in &trace {
        cache.access(k, desc);
    }
    assert_eq!(cache.resident_bytes(), cache.len() as u64 * MB);
    assert_eq!(cache.keys().count(), cache.len());
}

#[test]
fn an_expert_larger_than_the_whole_cache_is_refused() {
    let mut cache: ExpertCache<Vec<u8>> = ExpertCache::new(CacheConfig::with_capacity(MB));
    let k = ExpertKey::new(0, 0);
    let outcome = cache.admit(k, ExpertDesc::plain(4 * MB), vec![0u8; 8]);
    assert!(!outcome.admitted());
    assert!(!cache.contains(k));
    assert_eq!(cache.stats().rejected_oversized, 1);
    assert_eq!(cache.resident_bytes(), 0);
}

#[test]
fn everything_enters_the_window_first() {
    let mut cache = strata(64 * MB);
    let k = ExpertKey::new(1, 1);
    cache.get(k);
    cache.admit(k, ExpertDesc::plain(MB), ());
    assert_eq!(cache.region(k), Some(Region::Window));
}

#[test]
fn a_hot_expert_is_promoted_to_dequantised_residency() {
    let mut cache = strata(64 * MB);
    let hot = ExpertKey::new(0, 0);
    let desc = ExpertDesc {
        stored_bytes: MB,
        dequantized_bytes: 2 * MB,
    };

    // enough traffic to push it out of the window and past the promotion
    // threshold, with a little other traffic so it is not the only thing here
    for round in 0..64 {
        cache.access(hot, desc);
        cache.access(ExpertKey::new(0, 1 + round % 8), desc);
    }

    assert_eq!(cache.region(hot), Some(Region::Main));
    assert_eq!(cache.residency(hot), Some(Residency::Dequantized));
    assert!(cache.stats().promotions >= 1);
    assert!(cache.resident_bytes() <= cache.capacity_bytes());
}

#[test]
fn promotion_does_not_happen_when_it_would_not_fit() {
    // capacity is exactly two experts stored, so growing one to double size
    // would not fit and the promotion has to be declined
    let mut cache = strata(2 * MB);
    let desc = ExpertDesc {
        stored_bytes: MB,
        dequantized_bytes: 2 * MB,
    };
    for _ in 0..64 {
        cache.access(ExpertKey::new(0, 0), desc);
        cache.access(ExpertKey::new(0, 1), desc);
    }
    assert!(cache.resident_bytes() <= 2 * MB);
    assert_eq!(cache.stats().promotions, 0);
}

#[test]
fn bytes_missed_counts_every_byte_that_had_to_be_read() {
    let mut cache = strata(2 * MB);
    let desc = ExpertDesc::plain(MB);
    // four distinct experts in a cache that holds two, so every access after
    // the first pass is a miss and a re-read
    let keys: Vec<_> = (0..4).map(|e| ExpertKey::new(0, e)).collect();
    for _ in 0..10 {
        for &k in &keys {
            cache.access(k, desc);
        }
    }
    let s = cache.stats();
    assert_eq!(s.lookups(), 40);
    assert_eq!(
        s.bytes_missed,
        s.misses * MB,
        "every miss reads exactly one expert"
    );
    assert!(s.bytes_missed_per_lookup() > 0.0);
}

#[test]
fn removing_and_clearing_release_their_bytes() {
    let mut cache: ExpertCache<u32> = ExpertCache::new(CacheConfig::with_capacity(64 * MB));
    let desc = ExpertDesc::plain(MB);
    for e in 0..8 {
        let k = ExpertKey::new(0, e);
        cache.admit(k, desc, e);
        cache.get(k);
    }
    let before = cache.resident_bytes();
    assert_eq!(cache.remove(ExpertKey::new(0, 3)), Some(3));
    assert_eq!(cache.resident_bytes(), before - MB);
    assert!(cache.remove(ExpertKey::new(0, 3)).is_none());

    cache.clear();
    assert_eq!(cache.resident_bytes(), 0);
    assert!(cache.is_empty());
}

#[test]
fn resetting_stats_leaves_the_contents_alone() {
    let mut cache = strata(16 * MB);
    let trace = skewed_layer(32, 1_000, 9);
    let desc = ExpertDesc::plain(MB);
    for &k in &trace {
        cache.access(k, desc);
    }
    let held = cache.len();
    let bytes = cache.resident_bytes();

    cache.reset_stats();
    assert_eq!(cache.stats().lookups(), 0);
    assert!((cache.stats().hit_rate() - 0.0).abs() < f64::EPSILON);
    assert_eq!(
        cache.len(),
        held,
        "a warmup is not thrown away by resetting counters"
    );
    assert_eq!(cache.resident_bytes(), bytes);
}

#[test]
fn the_same_trace_twice_gives_the_same_answer() {
    let trace = skewed_layer(64, 5_000, 21);
    let a = run(&mut strata(16 * MB), &trace, MB);
    let b = run(&mut strata(16 * MB), &trace, MB);
    assert!(
        (a - b).abs() < f64::EPSILON,
        "replay must be reproducible: {a} vs {b}"
    );
}

#[test]
fn an_empty_trace_reports_nothing_rather_than_dividing_by_zero() {
    let cache = strata(MB);
    assert!((cache.stats().hit_rate() - 0.0).abs() < f64::EPSILON);
    assert!((cache.stats().bytes_missed_per_lookup() - 0.0).abs() < f64::EPSILON);
    assert!((cache.stats().churn_ratio() - 0.0).abs() < f64::EPSILON);

    let oracle = BeladyOracle::new(MB).run(&[], &HashMap::new());
    assert!((oracle.hit_rate() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn a_trace_that_fits_entirely_is_all_hits_after_the_first_pass() {
    let mut cache = strata(64 * MB);
    let desc = ExpertDesc::plain(MB);
    let keys: Vec<_> = (0..8).map(|e| ExpertKey::new(2, e)).collect();
    for &k in &keys {
        cache.access(k, desc);
    }
    cache.reset_stats();
    for _ in 0..20 {
        for &k in &keys {
            assert!(cache.access(k, desc), "{k} should be resident");
        }
    }
    assert!((cache.stats().hit_rate() - 1.0).abs() < f64::EPSILON);
    assert_eq!(cache.stats().bytes_missed, 0);
}

#[test]
fn expert_layer_pairs_are_tracked_separately() {
    // the same expert index in two layers must not share a cache entry, or
    // every statistic downstream is measuring the wrong thing
    let mut cache: ExpertCache<&str> = ExpertCache::new(CacheConfig::with_capacity(64 * MB));
    let desc = ExpertDesc::plain(MB);
    cache.admit(ExpertKey::new(3, 5), desc, "early");
    cache.admit(ExpertKey::new(30, 5), desc, "late");

    assert_eq!(cache.get(ExpertKey::new(3, 5)).copied(), Some("early"));
    assert_eq!(cache.get(ExpertKey::new(30, 5)).copied(), Some("late"));
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.resident_bytes(), 2 * MB);
}
