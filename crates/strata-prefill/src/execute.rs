//! running a layer in either order, and getting the same answer.

use crate::routing::LayerRouting;
use crate::schedule::LayerSchedule;
use strata_format::ExpertKey;

/// a block of token activations, `n_tokens` rows of `d_model`.
#[derive(Debug, Clone, PartialEq)]
pub struct Activations {
    d_model: usize,
    data: Vec<f32>,
}

impl Activations {
    /// wrap a flat row-major buffer.
    ///
    /// # Panics
    /// panics if the buffer is not a whole number of rows.
    #[must_use]
    pub fn new(d_model: usize, data: Vec<f32>) -> Self {
        assert!(d_model > 0, "d_model must be positive");
        assert_eq!(
            data.len() % d_model,
            0,
            "activations must be a whole number of rows"
        );
        Self { d_model, data }
    }

    /// zeros.
    #[must_use]
    pub fn zeros(n_tokens: usize, d_model: usize) -> Self {
        Self::new(d_model, vec![0.0; n_tokens * d_model])
    }

    /// rows.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        self.data.len() / self.d_model
    }

    /// width.
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// one row.
    #[must_use]
    pub fn row(&self, token: usize) -> &[f32] {
        &self.data[token * self.d_model..(token + 1) * self.d_model]
    }

    /// the flat buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// the ffn of one expert, applied to a stack of rows.
///
/// implementations are row independent: row `i` of the output depends only on
/// row `i` of the input. that is what makes reordering safe, and it is true of
/// every moe expert, which is a position-wise feed forward network.
pub trait ExpertFn {
    /// apply `key` to `rows` stacked input rows, writing `rows` output rows.
    fn apply(
        &mut self,
        key: ExpertKey,
        rows: usize,
        d_model: usize,
        input: &[f32],
        out: &mut [f32],
    );
}

/// the reference order: for each token, for each of its experts.
///
/// this is what a naive implementation does, and what every result here is
/// diffed against. it demands each expert once per token that wants it, so on a
/// long prefill it reads the same weights over and over. correct, and unusable
/// when the weights are on nvme.
pub fn run_token_major(
    routing: &LayerRouting,
    input: &Activations,
    op: &mut impl ExpertFn,
) -> Activations {
    let d = input.d_model();
    let mut out = Activations::zeros(input.n_tokens(), d);
    let mut scratch = vec![0.0f32; d];

    for token in 0..routing.n_tokens() {
        for slot in 0..routing.top_k() {
            let key = ExpertKey::new(routing.layer(), routing.expert_at(token, slot));
            let weight = routing.weight_at(token, slot);
            op.apply(key, 1, d, input.row(token), &mut scratch);

            let base = token * d;
            for (o, s) in out.data[base..base + d].iter_mut().zip(&scratch) {
                *o += weight * *s;
            }
        }
    }
    out
}

/// the strata order: for each expert in disk order, for every token that wants
/// it.
///
/// # why the answer is bit identical and not merely close
///
/// reordering the loops reorders the additions, and floating point addition is
/// not associative, so the obvious implementation of this returns slightly
/// different logits from the reference. that would make g5, which asks for
/// correctness verifiable by logit diff, fail for a reason that has nothing to
/// do with a bug and would hide the ones that do.
///
/// so contributions are not accumulated as they arrive. each one is written into
/// the slot it belongs to in a per token buffer, and the slots are summed in
/// order at the end, exactly as the token-major loop sums them. the cost is that
/// buffer, `n_tokens * top_k * d_model`, which is what
/// [`crate::block_size_for_budget`] exists to bound.
pub fn run_expert_major(
    schedule: &LayerSchedule,
    routing: &LayerRouting,
    input: &Activations,
    op: &mut impl ExpertFn,
) -> Activations {
    let d = input.d_model();
    let n = input.n_tokens();
    let k = routing.top_k();

    // one slot per token-slot, filled out of order, combined in order
    let mut contributions = vec![0.0f32; n * k * d];
    let mut gathered = Vec::new();
    let mut produced = Vec::new();

    for batch in &schedule.batches {
        let rows = batch.len();
        gathered.clear();
        gathered.reserve(rows * d);
        for ts in &batch.tokens {
            gathered.extend_from_slice(input.row(ts.token as usize));
        }
        produced.clear();
        produced.resize(rows * d, 0.0);

        // one call, one read, however many tokens wanted it
        op.apply(batch.key, rows, d, &gathered, &mut produced);

        for (r, ts) in batch.tokens.iter().enumerate() {
            let dst = ((ts.token as usize) * k + ts.slot as usize) * d;
            let src = r * d;
            for (c, p) in contributions[dst..dst + d]
                .iter_mut()
                .zip(&produced[src..src + d])
            {
                *c = ts.weight * *p;
            }
        }
    }

    let mut out = Activations::zeros(n, d);
    for token in 0..n {
        let base = token * d;
        // slots summed in index order, which is exactly the order the
        // token-major loop uses. this is what makes the two bit identical.
        for slot in 0..k {
            let src = (token * k + slot) * d;
            for (o, c) in out.data[base..base + d]
                .iter_mut()
                .zip(&contributions[src..src + d])
            {
                *o += *c;
            }
        }
    }
    out
}

/// counts how many times each expert was applied, for asserting g4.
#[derive(Debug, Default)]
pub struct CountingExpert<F> {
    inner: F,
    /// how many times each expert was applied.
    pub calls: std::collections::BTreeMap<ExpertKey, usize>,
    /// total rows processed, which is the work the gemm actually did.
    pub rows: usize,
}

impl<F> CountingExpert<F> {
    /// wrap an expert function with a call counter.
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            calls: std::collections::BTreeMap::new(),
            rows: 0,
        }
    }

    /// distinct experts touched.
    #[must_use]
    pub fn distinct(&self) -> usize {
        self.calls.len()
    }

    /// total applications, which token-major inflates and expert-major does not.
    #[must_use]
    pub fn total_calls(&self) -> usize {
        self.calls.values().sum()
    }
}

impl<F: ExpertFn> ExpertFn for CountingExpert<F> {
    fn apply(
        &mut self,
        key: ExpertKey,
        rows: usize,
        d_model: usize,
        input: &[f32],
        out: &mut [f32],
    ) {
        *self.calls.entry(key).or_insert(0) += 1;
        self.rows += rows;
        self.inner.apply(key, rows, d_model, input, out);
    }
}
