//! working out which bytes belong to which expert.
//!
//! the plan is built entirely from the header, before a single weight is read,
//! so a checkpoint that cannot be planned fails in milliseconds rather than
//! after writing a gigabyte of something wrong.

use crate::safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;
use strata_format::{ExpertKey, Precision};

/// why a checkpoint could not be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// no tensor matched any known mixture-of-experts naming convention.
    NoExpertTensors,
    /// the projections of one layer disagree about how many experts there are.
    ExpertCountMismatch {
        /// the layer where they disagreed.
        layer: u32,
        /// what was found.
        counts: Vec<u64>,
    },
    /// projections in the same layer are stored at different widths.
    MixedDtypes {
        /// the layer where they disagreed.
        layer: u32,
    },
    /// a dtype with no strata precision code.
    UnsupportedDtype(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoExpertTensors => write!(
                f,
                "no stacked expert tensors found. this converter recognises the \
                 `block_sparse_moe` and `mlp.experts` conventions; a new family \
                 needs its pattern adding to plan.rs"
            ),
            Self::ExpertCountMismatch { layer, counts } => write!(
                f,
                "layer {layer}: projections disagree about the expert count {counts:?}"
            ),
            Self::MixedDtypes { layer } => {
                write!(f, "layer {layer}: projections are stored at mixed widths")
            }
            Self::UnsupportedDtype(d) => write!(f, "dtype {d} has no strata precision code"),
        }
    }
}

impl std::error::Error for PlanError {}

/// one contiguous run of bytes to copy out of one source tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    /// name of the tensor in the source checkpoint.
    pub tensor: String,
    /// byte offset within that tensor.
    pub offset: u64,
    /// how many bytes.
    pub len: u64,
}

/// everything one expert-layer pair is made of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlan {
    /// which expert, in which layer.
    pub key: ExpertKey,
    /// the runs to concatenate, in order.
    ///
    /// order is the sorted tensor name, so the same model always produces the
    /// same payload. an engine reading this back has to agree with that order,
    /// which is why it is stable rather than whatever the header happened to
    /// list first.
    pub parts: Vec<Part>,
    /// total payload size.
    pub payload_len: u64,
}

/// every expert in a checkpoint, and the width they are stored at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPlan {
    /// the experts, ordered by layer then expert index.
    pub experts: Vec<ExpertPlan>,
    /// the precision every expert is stored at.
    pub precision: Precision,
    /// how many layers carry experts.
    pub layers: u32,
    /// experts per layer.
    pub experts_per_layer: u64,
    /// the projection tensors that were recognised, one layer's worth.
    pub projections: Vec<String>,
}

/// what a conversion did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertReport {
    /// how many expert-layer pairs were written.
    pub experts: usize,
    /// total payload bytes, before alignment padding.
    pub bytes_written: u64,
    /// the precision they were written at.
    pub precision: Precision,
}

/// one stacked expert tensor, as the header describes it.
#[derive(Debug, Clone)]
struct Stacked {
    /// name in the source checkpoint.
    tensor: String,
    /// size of the outermost dimension, which is the expert axis.
    experts: u64,
    /// bytes per expert along that axis.
    stride: u64,
    /// element width.
    dtype: Dtype,
}

/// layer index to projection name to the tensor holding it.
type ByLayer = BTreeMap<u32, BTreeMap<String, Stacked>>;

/// pull `layer` and the projection name out of a tensor name.
///
/// recognises the two conventions in the wild that stack experts on the
/// outermost dimension:
///
/// ```text
/// model.layers.3.block_sparse_moe.input_linear.weight   granite, mixtral-like
/// model.layers.3.mlp.experts.gate_up_proj                olmoe-like
/// ```
///
/// returns `None` for anything else, which is how attention and norms are
/// skipped without naming them.
fn classify(name: &str) -> Option<(u32, String)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_text, rest) = rest.split_once('.')?;
    let layer: u32 = layer_text.parse().ok()?;

    for marker in ["block_sparse_moe.", "mlp.experts.", "feed_forward.experts."] {
        if let Some(projection) = rest.strip_prefix(marker) {
            // the router is not an expert. it is one small matrix per layer that
            // stays resident, and pulling it into a paged expert payload would
            // mean faulting in the thing that decides what to fault in.
            //
            // matched on the whole first segment, not a prefix. `gate_up_proj`
            // is a fused expert projection and `gate.weight` is the router, so
            // a `starts_with("gate")` test silently drops half of every olmoe
            // expert and leaves a file that is the right shape and wrong.
            let head = projection.split('.').next().unwrap_or(projection);
            if head == "router" || head == "gate" {
                return None;
            }
            return Some((layer, projection.to_string()));
        }
    }
    None
}

/// build a conversion plan from a checkpoint's header alone.
///
/// # Errors
/// fails if nothing looks like a stacked expert tensor, or if the projections
/// of a layer disagree with each other.
pub fn plan(source: &SafeTensors) -> Result<ModelPlan, PlanError> {
    let mut by_layer: ByLayer = BTreeMap::new();

    for (name, info) in source.tensors() {
        let Some((layer, projection)) = classify(name) else {
            continue;
        };
        // a stacked expert tensor is at least [n_experts, ..]. a 1-d tensor
        // under this prefix is a bias or a norm and has no expert axis.
        if info.shape.len() < 2 {
            continue;
        }
        let Some(stride) = info.outer_stride() else {
            continue;
        };
        by_layer.entry(layer).or_default().insert(
            projection,
            Stacked {
                tensor: name.clone(),
                experts: info.shape[0],
                stride,
                dtype: info.dtype,
            },
        );
    }

    if by_layer.is_empty() {
        return Err(PlanError::NoExpertTensors);
    }

    let mut experts_out = Vec::new();
    let mut precision = None;
    let mut experts_per_layer = 0u64;
    let mut projections = Vec::new();

    for (&layer, projs) in &by_layer {
        // a layer only appears in the map because something was inserted into
        // it, so this cannot be empty. skipping rather than unwrapping keeps
        // that reasoning out of the runtime.
        let Some(head) = projs.values().next() else {
            continue;
        };

        let counts: Vec<u64> = projs.values().map(|s| s.experts).collect();
        let first = head.experts;
        if counts.iter().any(|n| *n != first) {
            return Err(PlanError::ExpertCountMismatch { layer, counts });
        }

        let dtype = head.dtype;
        if projs.values().any(|s| s.dtype != dtype) {
            return Err(PlanError::MixedDtypes { layer });
        }
        let p = dtype
            .precision()
            .ok_or_else(|| PlanError::UnsupportedDtype(format!("{dtype:?}")))?;
        precision = Some(p);

        experts_per_layer = first;
        if projections.is_empty() {
            projections = projs.keys().cloned().collect();
        }

        for expert in 0..first {
            let mut parts = Vec::with_capacity(projs.len());
            let mut payload_len = 0;
            // btreemap iteration is sorted by projection name, which is what
            // makes the payload order stable across runs and machines.
            for stacked in projs.values() {
                parts.push(Part {
                    tensor: stacked.tensor.clone(),
                    offset: expert * stacked.stride,
                    len: stacked.stride,
                });
                payload_len += stacked.stride;
            }
            experts_out.push(ExpertPlan {
                key: ExpertKey::new(layer, expert as u32),
                parts,
                payload_len,
            });
        }
    }

    // every layer in the map could still have been skipped above, so this is a
    // real case rather than an impossible one: it means nothing usable was
    // found, which is the same answer as finding nothing at all.
    let Some(precision) = precision else {
        return Err(PlanError::NoExpertTensors);
    };

    Ok(ModelPlan {
        experts: experts_out,
        precision,
        layers: by_layer.len() as u32,
        experts_per_layer,
        projections,
    })
}

#[cfg(test)]
mod tests {
    use super::classify;

    #[test]
    fn recognises_the_stacked_expert_conventions() {
        assert_eq!(
            classify("model.layers.3.block_sparse_moe.input_linear.weight"),
            Some((3, "input_linear.weight".to_string()))
        );
        assert_eq!(
            classify("model.layers.11.mlp.experts.gate_up_proj"),
            Some((11, "gate_up_proj".to_string()))
        );
    }

    #[test]
    fn skips_the_router_because_it_must_stay_resident() {
        assert_eq!(
            classify("model.layers.0.block_sparse_moe.router.layer.weight"),
            None
        );
        assert_eq!(classify("model.layers.0.mlp.experts.gate"), None);
    }

    #[test]
    fn skips_everything_that_is_not_an_expert() {
        for name in [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        ] {
            assert_eq!(classify(name), None, "{name} should not be an expert");
        }
    }
}
