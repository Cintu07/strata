//! the two claims of m4, stated as tests.
//!
//! g4: each expert is read at most once per layer per prefill, not once per
//! token.
//!
//! g5: the reordering does not change the answer. not "changes it only a
//! little": the outputs are compared bit for bit, because a tolerance here
//! would be a place for real bugs to hide.

use strata_format::{ExpertKey, LayoutReader, LayoutWriter, PlanOptions, Precision};
use strata_prefill::{
    Activations, CountingExpert, ExpertFn, LayerRouting, block_size_for_budget, run_expert_major,
    run_token_major, schedule_layer, schedule_layer_with,
};

const D_MODEL: usize = 8;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// a small dyadic rational, which is exactly representable in f32 so that
    /// any difference between two runs is a real difference and not rounding.
    fn weight(&mut self) -> f32 {
        (self.below(16) as f32 + 1.0) / 16.0
    }
}

/// a stand in for an expert ffn: row independent, deterministic, and cheap.
///
/// row independence is the property that makes reordering legal at all, and it
/// is true of every moe expert, which is a position-wise network.
struct MockExpert;

impl ExpertFn for MockExpert {
    fn apply(
        &mut self,
        key: ExpertKey,
        rows: usize,
        d_model: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        let scale = ((key.expert % 7) as f32 + 1.0) / 8.0;
        let bias = (key.layer % 3) as f32 / 4.0;
        for r in 0..rows {
            for i in 0..d_model {
                out[r * d_model + i] = input[r * d_model + i] * scale + bias;
            }
        }
    }
}

fn random_routing(
    layer: u32,
    n_tokens: usize,
    top_k: usize,
    n_experts: u32,
    seed: u64,
) -> LayerRouting {
    let mut rng = Rng::new(seed);
    let mut experts = Vec::with_capacity(n_tokens * top_k);
    let mut weights = Vec::with_capacity(n_tokens * top_k);
    for _ in 0..n_tokens {
        // distinct experts per token, as a real top-k is
        let mut chosen: Vec<u32> = Vec::new();
        while chosen.len() < top_k {
            let e = rng.below(u64::from(n_experts)) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        for e in chosen {
            experts.push(e);
            weights.push(rng.weight());
        }
    }
    LayerRouting::new(layer, top_k, experts, weights)
}

fn random_input(n_tokens: usize, seed: u64) -> Activations {
    let mut rng = Rng::new(seed);
    let data = (0..n_tokens * D_MODEL).map(|_| rng.weight()).collect();
    Activations::new(D_MODEL, data)
}

// ------------------------------------------------------------------ g5

#[test]
fn expert_major_is_bit_identical_to_token_major() {
    for (tokens, top_k, experts, seed) in [
        (64usize, 2usize, 16u32, 1u64),
        (129, 4, 32, 2),
        (7, 8, 8, 3),
        (1, 1, 4, 4),
    ] {
        let routing = random_routing(0, tokens, top_k, experts, seed);
        let input = random_input(tokens, seed + 100);
        let schedule = schedule_layer_with(&routing, |k| Some(u64::from(k.expert) * 4096));

        let reference = run_token_major(&routing, &input, &mut MockExpert);
        let reordered = run_expert_major(&schedule, &routing, &input, &mut MockExpert);

        assert_eq!(
            reference.as_slice(),
            reordered.as_slice(),
            "reordering changed the answer for {tokens} tokens, top-{top_k} of {experts}"
        );
    }
}

#[test]
fn the_answer_does_not_depend_on_the_disk_order_chosen() {
    let routing = random_routing(0, 96, 3, 24, 11);
    let input = random_input(96, 12);

    let forward = schedule_layer_with(&routing, |k| Some(u64::from(k.expert) * 4096));
    let reversed = schedule_layer_with(&routing, |k| Some(u64::from(100 - k.expert) * 4096));

    let a = run_expert_major(&forward, &routing, &input, &mut MockExpert);
    let b = run_expert_major(&reversed, &routing, &input, &mut MockExpert);
    assert_eq!(
        a.as_slice(),
        b.as_slice(),
        "the layout must not change the logits"
    );
}

// ------------------------------------------------------------------ g4

#[test]
fn each_expert_is_applied_exactly_once_per_layer() {
    let routing = random_routing(5, 512, 4, 32, 7);
    let input = random_input(512, 8);
    let schedule = schedule_layer_with(&routing, |k| Some(u64::from(k.expert) * 4096));

    let mut counted = CountingExpert::new(MockExpert);
    run_expert_major(&schedule, &routing, &input, &mut counted);

    assert_eq!(
        counted.total_calls(),
        counted.distinct(),
        "an expert was applied twice"
    );
    assert_eq!(counted.distinct(), schedule.reads);
    assert_eq!(
        counted.rows, schedule.assignments,
        "every token-slot must be served exactly once"
    );
}

#[test]
fn token_major_pays_for_the_same_expert_over_and_over() {
    let routing = random_routing(0, 512, 4, 32, 9);
    let input = random_input(512, 10);
    let schedule = schedule_layer_with(&routing, |k| Some(u64::from(k.expert) * 4096));

    let mut token_major = CountingExpert::new(MockExpert);
    run_token_major(&routing, &input, &mut token_major);

    let mut expert_major = CountingExpert::new(MockExpert);
    run_expert_major(&schedule, &routing, &input, &mut expert_major);

    assert_eq!(
        token_major.total_calls(),
        512 * 4,
        "one call per token-slot"
    );
    assert_eq!(
        expert_major.total_calls(),
        32,
        "one call per distinct expert"
    );
    assert!(
        token_major.total_calls() > expert_major.total_calls() * 50,
        "the whole point is that this ratio is large: {} against {}",
        token_major.total_calls(),
        expert_major.total_calls()
    );
    // the same total work reaches the gemm either way, in far fewer calls
    assert_eq!(token_major.rows, expert_major.rows);
    assert!(
        schedule.mean_batch() > 50.0,
        "batches were {:.1} rows",
        schedule.mean_batch()
    );
}

// ------------------------------------------------------------- scheduling

#[test]
fn batches_are_emitted_in_disk_order() {
    let routing = random_routing(0, 200, 3, 20, 13);
    // deliberately the inverse of index order, so index order would fail
    let schedule = schedule_layer_with(&routing, |k| Some(u64::from(50 - k.expert) * 4096));

    let offsets: Vec<u64> = schedule
        .batches
        .iter()
        .map(|b| u64::from(50 - b.key.expert))
        .collect();
    let mut sorted = offsets.clone();
    sorted.sort_unstable();
    assert_eq!(offsets, sorted, "reads must sweep forward through the file");
}

#[test]
fn every_assignment_is_accounted_for_exactly_once() {
    let routing = random_routing(3, 128, 4, 16, 17);
    let schedule = schedule_layer_with(&routing, |k| Some(u64::from(k.expert)));

    let mut seen = vec![false; routing.n_tokens() * routing.top_k()];
    for batch in &schedule.batches {
        for ts in &batch.tokens {
            let idx = ts.token as usize * routing.top_k() + ts.slot as usize;
            assert!(
                !seen[idx],
                "token {} slot {} appeared twice",
                ts.token, ts.slot
            );
            seen[idx] = true;
            assert_eq!(
                routing.expert_at(ts.token as usize, ts.slot as usize),
                batch.key.expert,
                "a token-slot was filed under the wrong expert"
            );
        }
    }
    assert!(seen.iter().all(|&s| s), "some token-slot was dropped");
}

#[test]
fn an_expert_missing_from_the_layout_is_dropped_visibly() {
    let routing = random_routing(0, 32, 2, 8, 19);
    // pretend experts 0 and 1 were never written to the file
    let schedule = schedule_layer_with(&routing, |k| (k.expert >= 2).then(|| u64::from(k.expert)));

    assert!(schedule.batches.iter().all(|b| b.key.expert >= 2));
    assert!(
        schedule.assignments < 32 * 2,
        "the caller can see the shortfall by comparing against n_tokens * top_k"
    );
}

#[test]
fn the_block_size_respects_the_memory_budget() {
    // 4 bytes an element, top-4, d_model 4096: 64kb of contribution buffer per
    // token, so a 64mb budget is a thousand tokens
    let n = block_size_for_budget(64 << 20, 4, 4096, 4);
    assert_eq!(n, 1024);
    assert!(n * 4 * 4096 * 4 <= 64 << 20);

    // and a budget too small for even one token still yields one, rather than
    // refusing to run
    assert_eq!(block_size_for_budget(1, 4, 4096, 4), 1);
}

#[test]
fn an_empty_block_schedules_nothing() {
    let routing = LayerRouting::new(0, 2, vec![], vec![]);
    assert!(routing.is_empty());
    let schedule = schedule_layer_with(&routing, |_| Some(0));
    assert_eq!(schedule.reads, 0);
    assert_eq!(schedule.assignments, 0);
    assert_eq!(schedule.reads_saved(), 0);
    assert!((schedule.mean_batch() - 0.0).abs() < f64::EPSILON);
}

// ------------------------------------------------- end to end with the file

/// the sweep, through the real layout file and the real read planner.
///
/// this is the claim that matters for io: a whole prefill layer becomes a small
/// number of large sequential transfers rather than one request per token-slot.
#[test]
fn a_prefill_layer_becomes_a_short_sweep_of_the_file() {
    let n_experts = 32u32;
    let path = std::env::temp_dir().join(format!("strata-prefill-{}.strata", std::process::id()));

    let mut w = LayoutWriter::create(&path, "prefill-test").unwrap();
    for e in 0..n_experts {
        w.push_expert(
            ExpertKey::new(0, e),
            Precision::Q4,
            &vec![(e % 251) as u8; 64 * 1024],
        )
        .unwrap();
    }
    w.finish().unwrap();
    let reader = LayoutReader::open(&path).unwrap();

    // 512 tokens, top-4, so 2048 token-slots over 32 experts
    let routing = random_routing(0, 512, 4, n_experts, 23);
    let schedule = schedule_layer(&routing, &reader);

    assert_eq!(schedule.assignments, 2048);
    assert_eq!(schedule.reads, 32, "g4: one read per expert per layer");

    let plan = schedule.read_plan(&reader, PlanOptions::default()).unwrap();
    assert_eq!(
        plan.requests.len(),
        1,
        "32 adjacent experts of 64kb coalesce into a single transfer, \
         against 2048 requests token-major"
    );
    assert!(
        (plan.overfetch_ratio() - 1.0).abs() < 1e-9,
        "and nothing extra was read"
    );

    // the bytes are real, so the sweep can actually be executed
    let bytes = reader.execute(&plan.requests[0]).unwrap();
    assert_eq!(bytes.len() as u64, plan.requests[0].len);

    std::fs::remove_file(&path).unwrap();
}
