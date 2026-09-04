//! the layout pass has to earn its place: an order produced from a profile
//! must capture more co-activation weight than index order, and that has to
//! turn into fewer reads through the actual planner rather than only looking
//! good in a metric of its own devising.

use std::collections::HashMap;
use strata_format::{
    CoactivationEdge, ExpertKey, LayoutReader, LayoutWriter, PlanOptions, Precision,
};
use strata_layout::{CoactivationProfile, capture_ratio, order_layer, plan_layout};

const EXPERT_BYTES: u64 = 64 * 1024;

/// deterministic generator so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// a layer whose experts fall into tight groups that fire together, with the
/// group members deliberately scattered across the index space so that index
/// order is the wrong answer.
fn grouped_profile(n_experts: u32, group_size: u32, tokens: u32, seed: u64) -> CoactivationProfile {
    let mut rng = Rng::new(seed);
    let n_groups = n_experts / group_size;
    let mut profile = CoactivationProfile::new();
    for e in 0..n_experts {
        profile.declare(0, e);
    }
    for _ in 0..tokens {
        let g = rng.below(u64::from(n_groups)) as u32;
        // members of group g are g, g + n_groups, g + 2*n_groups, ...
        let members: Vec<u32> = (0..group_size).map(|i| g + i * n_groups).collect();
        profile.observe(0, &members);
    }
    profile
}

fn uniform_sizes(n: u32) -> HashMap<u32, u64> {
    (0..n).map(|e| (e, EXPERT_BYTES)).collect()
}

#[test]
fn co_activation_order_captures_more_weight_than_index_order() {
    let n = 64u32;
    let profile = grouped_profile(n, 4, 4_000, 5);
    let edges = profile.layer_edges(0, 0.0);
    let sizes = uniform_sizes(n);

    let by_index: Vec<u32> = (0..n).collect();
    let by_coactivation = order_layer(&profile.experts_in(0), &edges);

    // window of zero means strictly adjacent, the hardest version of the test
    let indexed = capture_ratio(&by_index, &sizes, &edges, 0);
    let ordered = capture_ratio(&by_coactivation, &sizes, &edges, 0);

    assert!(
        ordered > indexed + 0.3,
        "ordering should capture much more weight: {ordered:.3} against {indexed:.3}"
    );
}

#[test]
fn ordering_is_a_permutation_and_loses_nothing() {
    let n = 64u32;
    let profile = grouped_profile(n, 4, 2_000, 9);
    let edges = profile.layer_edges(0, 0.0);
    let order = order_layer(&profile.experts_in(0), &edges);

    let mut sorted = order.clone();
    sorted.sort_unstable();
    let expected: Vec<u32> = (0..n).collect();
    assert_eq!(sorted, expected, "every expert must appear exactly once");
}

#[test]
fn an_expert_the_corpus_never_touched_is_still_placed() {
    let mut profile = CoactivationProfile::new();
    for _ in 0..50 {
        profile.observe(0, &[1, 2]);
    }
    // expert 7 exists in the model but the profiling corpus never routed to it
    profile.declare(0, 7);

    let order = order_layer(&profile.experts_in(0), &profile.layer_edges(0, 0.0));
    assert!(
        order.contains(&7),
        "a cold expert is still an expert: {order:?}"
    );
    assert_eq!(order.len(), 3);
}

#[test]
fn the_same_profile_always_produces_the_same_order() {
    let profile = grouped_profile(48, 3, 3_000, 17);
    let edges = profile.layer_edges(0, 0.0);
    let a = order_layer(&profile.experts_in(0), &edges);
    let b = order_layer(&profile.experts_in(0), &edges);
    assert_eq!(
        a, b,
        "a layout that shuffles makes every comparison meaningless"
    );
}

#[test]
fn a_profile_with_no_structure_is_handled_without_pretending_otherwise() {
    let mut rng = Rng::new(3);
    let mut profile = CoactivationProfile::new();
    for e in 0..32 {
        profile.declare(0, e);
    }
    for _ in 0..2_000 {
        let picks: Vec<u32> = (0..4).map(|_| rng.below(32) as u32).collect();
        profile.observe(0, &picks);
    }
    let edges = profile.layer_edges(0, 0.0);
    let order = order_layer(&profile.experts_in(0), &edges);
    assert_eq!(order.len(), 32);

    // with uniform routing there is no structure to capture, and the pass
    // should not claim to have found any
    let sizes = uniform_sizes(32);
    let ratio = capture_ratio(&order, &sizes, &edges, 0);
    assert!(
        ratio < 0.2,
        "no structure should mean little capture, got {ratio:.3}"
    );
}

#[test]
fn duplicate_experts_in_one_observation_do_not_inflate_the_counts() {
    let mut a = CoactivationProfile::new();
    let mut b = CoactivationProfile::new();
    for _ in 0..10 {
        a.observe(0, &[1, 2, 3]);
        b.observe(0, &[1, 2, 2, 3, 3]);
    }
    assert_eq!(a.layer_edges(0, 0.0), b.layer_edges(0, 0.0));
    assert_eq!(
        a.hit_count(ExpertKey::new(0, 2)),
        b.hit_count(ExpertKey::new(0, 2))
    );
}

#[test]
fn weights_are_joint_probabilities_and_the_threshold_drops_rare_pairs() {
    let mut profile = CoactivationProfile::new();
    for _ in 0..100 {
        profile.observe(0, &[0, 1]);
    }
    profile.observe(0, &[0, 9]);

    let all = profile.layer_edges(0, 0.0);
    let strong = profile.layer_edges(0, 0.05);

    let common = all.iter().find(|e| (e.a, e.b) == (0, 1)).expect("pair 0-1");
    assert!(
        (common.weight - 100.0 / 101.0).abs() < 1e-5,
        "weight was {}",
        common.weight
    );

    assert!(
        all.len() > strong.len(),
        "the threshold should have dropped the rare pair"
    );
    assert!(strong.iter().all(|e| (e.a, e.b) == (0, 1)));
}

#[test]
fn layers_stay_grouped_and_in_order_in_the_full_plan() {
    let mut profile = CoactivationProfile::new();
    for layer in [0u32, 1, 2] {
        for _ in 0..20 {
            profile.observe(layer, &[0, 1]);
            profile.observe(layer, &[2, 3]);
        }
    }
    let plan = plan_layout(&profile, 0.0);
    assert_eq!(plan.len(), 12);

    let layers: Vec<u32> = plan.iter().map(|k| k.layer).collect();
    let mut expected_sorted = layers.clone();
    expected_sorted.sort_unstable();
    assert_eq!(
        layers, expected_sorted,
        "a prefill sweep must move forward through the file"
    );
}

/// the end to end claim: the ordering pass, written through the real layout
/// file, read back through the real planner, produces fewer requests.
///
/// the metric in `capture_ratio` and the behaviour of the planner are two
/// separate pieces of code, and it would be entirely possible for the first to
/// improve while the second did not.
#[test]
fn the_ordering_turns_into_fewer_reads_through_the_real_planner() {
    let n = 64u32;
    let profile = grouped_profile(n, 4, 4_000, 23);
    let edges = profile.layer_edges(0, 0.0);
    let by_coactivation = order_layer(&profile.experts_in(0), &edges);
    let by_index: Vec<u32> = (0..n).collect();

    // a token routes to one whole group, which is the access pattern the
    // layout was profiled on
    let group = [7u32, 7 + 16, 7 + 32, 7 + 48];
    let wanted: Vec<ExpertKey> = group.iter().map(|&e| ExpertKey::new(0, e)).collect();

    let mut requests = Vec::new();
    for (tag, order) in [("index", &by_index), ("coactivation", &by_coactivation)] {
        let path =
            std::env::temp_dir().join(format!("strata-layout-{tag}-{}.strata", std::process::id()));
        let mut w = LayoutWriter::create(&path, "layout-test").unwrap();
        for &e in order {
            w.push_expert(
                ExpertKey::new(0, e),
                Precision::Q4,
                &vec![(e % 251) as u8; EXPERT_BYTES as usize],
            )
            .unwrap();
        }
        w.set_coactivation(
            edges
                .iter()
                .map(|e| CoactivationEdge {
                    layer: 0,
                    a: e.a,
                    b: e.b,
                    weight: e.weight,
                })
                .collect(),
        );
        w.finish().unwrap();

        let r = LayoutReader::open(&path).unwrap();
        // no bridging at all, so the only thing being measured is adjacency
        let plan = r.plan_reads(&wanted, PlanOptions::no_overfetch()).unwrap();
        requests.push(plan.requests.len());
        std::fs::remove_file(&path).unwrap();
    }

    let (indexed, ordered) = (requests[0], requests[1]);
    assert_eq!(
        indexed, 4,
        "in index order the group is scattered, so four reads"
    );
    assert_eq!(
        ordered, 1,
        "in co-activation order the group is contiguous, so one read"
    );
}
