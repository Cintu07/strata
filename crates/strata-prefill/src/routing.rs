//! routing for one layer of a prefill block, and its inversion.

use std::collections::HashMap;
use strata_format::ExpertKey;

/// which experts each token in a block routed to at one layer.
///
/// stored flat as `[token * top_k + slot]` rather than as a vector of vectors,
/// because the router produces a dense top-k and the whole point of this crate
/// is to walk it in a different order without copying it.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRouting {
    layer: u32,
    top_k: usize,
    experts: Vec<u32>,
    weights: Vec<f32>,
}

impl LayerRouting {
    /// build from the router's output.
    ///
    /// # Panics
    /// panics if the two arrays disagree in length or do not divide by `top_k`.
    /// a mismatch here silently misattributes every expert to the wrong token,
    /// which produces plausible nonsense rather than an error.
    #[must_use]
    pub fn new(layer: u32, top_k: usize, experts: Vec<u32>, weights: Vec<f32>) -> Self {
        assert!(top_k > 0, "top_k must be positive");
        assert_eq!(experts.len(), weights.len(), "one weight per expert slot");
        assert_eq!(
            experts.len() % top_k,
            0,
            "expert list must be a whole number of tokens"
        );
        Self {
            layer,
            top_k,
            experts,
            weights,
        }
    }

    /// the layer this routing belongs to.
    #[must_use]
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    /// experts routed per token.
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    /// tokens in this block.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        self.experts.len() / self.top_k
    }

    /// whether the block is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }

    /// the expert in one slot of one token.
    #[must_use]
    pub fn expert_at(&self, token: usize, slot: usize) -> u32 {
        self.experts[token * self.top_k + slot]
    }

    /// the router weight for one slot of one token.
    #[must_use]
    pub fn weight_at(&self, token: usize, slot: usize) -> f32 {
        self.weights[token * self.top_k + slot]
    }

    /// invert token to expert into expert to token.
    ///
    /// this is the whole trick. token-major order asks for an expert once per
    /// token that wants it, which on a 4096 token prefill means reading the same
    /// 50mb of weights hundreds of times. expert-major asks once and applies the
    /// result to every token at once, which satisfies g4 and turns the ffn into
    /// a batched gemm with real arithmetic intensity instead of a stream of
    /// tiny ones.
    #[must_use]
    pub fn invert(&self) -> Vec<ExpertBatch> {
        let mut by_expert: HashMap<u32, Vec<TokenSlot>> = HashMap::new();
        for token in 0..self.n_tokens() {
            for slot in 0..self.top_k {
                by_expert
                    .entry(self.expert_at(token, slot))
                    .or_default()
                    .push(TokenSlot {
                        token: token as u32,
                        slot: slot as u8,
                        weight: self.weight_at(token, slot),
                    });
            }
        }
        let mut batches: Vec<ExpertBatch> = by_expert
            .into_iter()
            .map(|(expert, mut tokens)| {
                // sorted so a batch is deterministic and so the gemm walks the
                // activation buffer forwards
                tokens.sort_unstable_by_key(|t| (t.token, t.slot));
                ExpertBatch {
                    key: ExpertKey::new(self.layer, expert),
                    tokens,
                }
            })
            .collect();
        batches.sort_unstable_by_key(|b| b.key);
        batches
    }
}

/// one token's claim on an expert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSlot {
    /// index within the prefill block.
    pub token: u32,
    /// which of the token's top-k slots this is.
    ///
    /// carried so that contributions computed out of order can be recombined in
    /// slot order. without it, expert-major and token-major would sum a token's
    /// experts in different orders and produce different floating point results,
    /// which would make g5's logit diff against the reference fail for a reason
    /// that has nothing to do with a bug.
    pub slot: u8,
    /// the router's weight for this token and expert.
    pub weight: f32,
}

/// one expert and every token in the block that wants it.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBatch {
    /// the expert-layer pair to read.
    pub key: ExpertKey,
    /// the tokens to apply it to, sorted by token then slot.
    pub tokens: Vec<TokenSlot>,
}

impl ExpertBatch {
    /// how many token-slots this batch serves, which is the m dimension of the
    /// gemm it becomes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// whether no token wants this expert.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
