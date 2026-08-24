//! turn a safetensors model into a strata layout file.
//!
//! this is the piece of plumbing that was missing for the whole of m0. the
//! layout writer takes bytes, the storage tier reads bytes, and until now
//! nothing turned a real model into those bytes, so every number the rust half
//! of this project produced came from synthetic payloads.
//!
//! # what an expert is, on disk
//!
//! a mixture-of-experts checkpoint does not store experts separately. it stores
//! one stacked tensor per projection per layer, with the expert index as the
//! outermost dimension:
//!
//! ```text
//! model.layers.3.block_sparse_moe.input_linear.weight   [32, 1024, 1024] BF16
//! model.layers.3.block_sparse_moe.output_linear.weight  [32, 1024,  512] BF16
//! ```
//!
//! so expert 5 of layer 3 is row 5 of each of those, and it is not contiguous
//! with itself: its two halves sit megabytes apart in the file, and the same
//! expert in the next layer is further away still. that is exactly the layout
//! that makes an offloading engine issue two scattered reads per expert per
//! token.
//!
//! the conversion is therefore the whole point rather than a format change: it
//! gathers the projections belonging to one expert-layer pair, concatenates
//! them in a fixed order, and hands the result to the layout writer, which
//! places it contiguously and 4kb aligned. after that, one expert is one read.
//!
//! # what it does not do
//!
//! it does not quantise, and it does not convert anything but the experts.
//! attention, embeddings and norms stay in the source checkpoint, because they
//! are resident in ram in every design this project is aimed at and moving them
//! would buy nothing. the layout file is the part that gets paged.

pub mod json;
pub mod plan;
pub mod safetensors;

pub use plan::{ConvertReport, ExpertPlan, ModelPlan, PlanError};
pub use safetensors::{Dtype, Error as SafeTensorsError, SafeTensors, TensorInfo};

use std::collections::HashMap;
use std::path::Path;
use strata_format::{CoactivationEdge, ExpertKey, LayoutWriter, RouteTrace};
use strata_layout::CoactivationProfile;

/// anything that stops a conversion.
#[derive(Debug)]
pub enum Error {
    /// reading the source checkpoint.
    Source(safetensors::Error),
    /// the checkpoint does not look like a model this can plan.
    Plan(PlanError),
    /// writing the layout file.
    Format(strata_format::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(e) => write!(f, "{e}"),
            Self::Plan(e) => write!(f, "{e}"),
            Self::Format(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<safetensors::Error> for Error {
    fn from(e: safetensors::Error) -> Self {
        Self::Source(e)
    }
}

impl From<PlanError> for Error {
    fn from(e: PlanError) -> Self {
        Self::Plan(e)
    }
}

impl From<strata_format::Error> for Error {
    fn from(e: strata_format::Error) -> Self {
        Self::Format(e)
    }
}

/// convert every expert in `plan` into a layout file at `out`.
///
/// experts are written in the order `plan` lists them, so an ordering computed
/// by `strata-layout` can be applied simply by sorting the plan before calling
/// this.
///
/// memory use is one expert, not one model. the largest buffer allocated is the
/// payload of the biggest expert-layer pair, which for granite is 3 mib against
/// a 2.7gb checkpoint.
///
/// # Errors
/// fails on io error, a header that disagrees with itself, or a duplicate key.
pub fn convert(
    source: &mut SafeTensors,
    plan: &ModelPlan,
    out: impl AsRef<Path>,
    model_id: &str,
) -> Result<ConvertReport, Error> {
    convert_with_edges(source, plan, out, model_id, &[])
}

/// convert, and attach a measured co-activation graph to the layout file.
///
/// the graph is what lets a reader coalesce a scattered want-set without
/// re-deriving which experts fire together, so it belongs in the file next to
/// the ordering it justifies rather than in a sidecar that can go missing.
///
/// # Errors
/// fails on io error, a header that disagrees with itself, or a duplicate key.
pub fn convert_with_edges(
    source: &mut SafeTensors,
    plan: &ModelPlan,
    out: impl AsRef<Path>,
    model_id: &str,
    edges: &[CoactivationEdge],
) -> Result<ConvertReport, Error> {
    let mut writer = LayoutWriter::create(out, model_id)?;
    if !edges.is_empty() {
        writer.set_coactivation(edges.to_vec());
    }
    let mut payload = Vec::new();
    let mut bytes_written = 0u64;

    for expert in &plan.experts {
        payload.clear();
        payload.resize(expert.payload_len as usize, 0);

        let mut at = 0usize;
        for part in &expert.parts {
            let end = at + part.len as usize;
            source.read_into(&part.tensor, part.offset, &mut payload[at..end])?;
            at = end;
        }
        debug_assert_eq!(at, payload.len());

        writer.push_expert(expert.key, plan.precision, &payload)?;
        bytes_written += expert.payload_len;
    }

    let experts = writer.len();
    writer.finish()?;

    Ok(ConvertReport {
        experts,
        bytes_written,
        precision: plan.precision,
    })
}

/// the expert-layer pairs a plan covers, for reporting.
#[must_use]
pub fn keys(plan: &ModelPlan) -> Vec<ExpertKey> {
    plan.experts.iter().map(|e| e.key).collect()
}

/// build a co-activation profile from an observed routing trace.
///
/// every expert is declared before the trace is walked, so an expert the corpus
/// never routed to still gets a place in the file. leaving it out would make
/// the layout depend on corpus coverage, and a model that silently loses the
/// experts a short corpus missed is a worse failure than a suboptimal order.
#[must_use]
pub fn profile_from_trace(trace: &RouteTrace, plan: &ModelPlan) -> CoactivationProfile {
    let mut profile = CoactivationProfile::new();
    for expert in &plan.experts {
        profile.declare(expert.key.layer, expert.key.expert);
    }

    let mut selection: Vec<u32> = Vec::with_capacity(trace.top_k);
    for token in 0..trace.n_tokens {
        for layer in 0..trace.n_layers {
            selection.clear();
            selection.extend(trace.selection(token, layer).iter().map(|k| k.expert));
            profile.observe(layer as u32, &selection);
        }
    }
    profile
}

/// reorder a plan so experts land on disk in co-activation order.
///
/// experts the ordering does not mention keep their original relative position
/// at the end of their layer, so the result is always a permutation of the
/// input rather than a subset of it.
///
/// # why the order is the point
///
/// two experts that fire together on the same token are two reads if they sit
/// far apart and, if the gap is small enough for the planner to bridge, one
/// read if they are adjacent. decision 0009 measured that the gain is in
/// request count rather than bytes, so this matters most for small experts and
/// least for large ones.
pub fn reorder(plan: &mut ModelPlan, order: &[ExpertKey]) {
    let mut rank: HashMap<ExpertKey, usize> = HashMap::with_capacity(order.len());
    for (i, key) in order.iter().enumerate() {
        rank.insert(*key, i);
    }

    // an unmentioned expert sorts after every mentioned one, inside its layer.
    let unranked = order.len();
    plan.experts.sort_by_key(|e| {
        (
            e.key.layer,
            rank.get(&e.key).copied().unwrap_or(unranked),
            e.key.expert,
        )
    });
}
