//! ordering the batches so the file is swept once.

use crate::routing::{ExpertBatch, LayerRouting};
use strata_format::{ExpertKey, LayoutReader, PlanOptions, ReadPlan};

/// how a prefill layer is going to be executed.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSchedule {
    /// batches in the order their experts sit on disk.
    pub batches: Vec<ExpertBatch>,
    /// token-slots served, which should equal `n_tokens * top_k`.
    pub assignments: usize,
    /// distinct experts, which is the number of reads this layer will cost.
    pub reads: usize,
}

impl LayerSchedule {
    /// reads saved against doing it token-major.
    ///
    /// token-major asks for an expert once per token that wants it. this is the
    /// difference, and on a real prefill it is most of the io.
    #[must_use]
    pub fn reads_saved(&self) -> usize {
        self.assignments.saturating_sub(self.reads)
    }

    /// mean tokens per expert read, which is the batch size the gemm gets.
    #[must_use]
    pub fn mean_batch(&self) -> f64 {
        if self.reads == 0 {
            return 0.0;
        }
        self.assignments as f64 / self.reads as f64
    }
}

/// order a layer's batches by where their experts sit in the layout file.
///
/// # why disk order and not router order
///
/// the read planner coalesces requests that are near each other, and it can only
/// do that if they arrive in ascending offset order. issuing in router order
/// turns a file that was carefully laid out by co-activation back into random
/// access, which measurement puts at roughly a hundredth of the throughput of
/// large reads. sorting here is what makes the layout pass mean anything at
/// execution time.
///
/// experts the layout file does not contain are dropped, and the caller can spot
/// that by comparing `assignments` against `n_tokens * top_k`.
#[must_use]
pub fn schedule_layer(routing: &LayerRouting, reader: &LayoutReader) -> LayerSchedule {
    let data_off = reader.header().data_off;
    let mut batches = routing.invert();

    batches.retain(|b| reader.entry(b.key).is_some());
    batches.sort_by_key(|b| {
        reader
            .entry(b.key)
            .map_or(u64::MAX, |e| e.file_offset(data_off))
    });

    let assignments = batches.iter().map(ExpertBatch::len).sum();
    let reads = batches.len();
    LayerSchedule {
        batches,
        assignments,
        reads,
    }
}

/// order batches by an arbitrary offset lookup, for callers without a file.
///
/// the tests use this, and so does anything scheduling against a layout that is
/// still being planned.
#[must_use]
pub fn schedule_layer_with(
    routing: &LayerRouting,
    offset_of: impl Fn(ExpertKey) -> Option<u64>,
) -> LayerSchedule {
    let mut batches = routing.invert();
    batches.retain(|b| offset_of(b.key).is_some());
    batches.sort_by_key(|b| offset_of(b.key).unwrap_or(u64::MAX));

    let assignments = batches.iter().map(ExpertBatch::len).sum();
    let reads = batches.len();
    LayerSchedule {
        batches,
        assignments,
        reads,
    }
}

impl LayerSchedule {
    /// the read plan for this layer, in one sweep.
    ///
    /// every expert appears once, in disk order, so the planner sees exactly the
    /// pattern it can coalesce. this is g4 stated as a call: each expert read at
    /// most once per layer per prefill, not once per token.
    ///
    /// # Errors
    /// fails if an expert in the schedule is not in the layout file.
    pub fn read_plan(
        &self,
        reader: &LayoutReader,
        opts: PlanOptions,
    ) -> strata_format::Result<ReadPlan> {
        let keys: Vec<ExpertKey> = self.batches.iter().map(|b| b.key).collect();
        reader.plan_reads(&keys, opts)
    }
}

/// split a prefill into token blocks that fit a memory budget.
///
/// expert-major execution has to hold a contribution buffer for every
/// token-slot in flight, because a token's experts arrive in whatever order the
/// disk hands them over and cannot be summed until they are all present. that
/// buffer is the cost of the reordering, and chunking is what bounds it.
///
/// returns the block size in tokens, at least one. a budget too small for even a
/// single token still yields one, because the alternative is refusing to run,
/// and one token at a time is exactly the token-major behaviour this is trying
/// to avoid rather than an impossibility.
#[must_use]
pub fn block_size_for_budget(
    budget_bytes: u64,
    top_k: usize,
    d_model: usize,
    bytes_per_element: usize,
) -> usize {
    let per_token = (top_k * d_model * bytes_per_element) as u64;
    if per_token == 0 {
        return 1;
    }
    (budget_bytes / per_token).max(1) as usize
}
