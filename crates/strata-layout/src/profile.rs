//! accumulating the co-activation graph from observed routing.

use std::collections::{BTreeMap, BTreeSet};
use strata_format::{CoactivationEdge, ExpertKey};

/// counts of which experts were routed together, per layer.
///
/// fed from a trace of real routing decisions. the output is the graph the
/// layout pass orders the file by and the graph the runtime consults when a
/// miss is about to become a read anyway.
///
/// ordered maps throughout, so that two runs over the same trace produce
/// byte identical output. a layout pass that shuffles between runs makes every
/// before and after measurement meaningless.
#[derive(Debug, Default, Clone)]
pub struct CoactivationProfile {
    /// tokens observed per layer, the denominator of every weight.
    tokens: BTreeMap<u32, u64>,
    /// `(layer, a, b)` with `a < b`, counting tokens that routed to both.
    pairs: BTreeMap<(u32, u32, u32), u64>,
    /// experts seen at all, so a layer with an expert nobody pairs with still
    /// gets it placed.
    experts: BTreeMap<u32, BTreeSet<u32>>,
    /// how often each expert was routed, for the report and for tie breaking.
    hits: BTreeMap<(u32, u32), u64>,
}

impl CoactivationProfile {
    /// an empty profile.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// record the experts one token routed to in one layer.
    ///
    /// duplicates in `experts` are ignored, so a caller that hands over a
    /// top-k list with a repeat does not skew the counts.
    pub fn observe(&mut self, layer: u32, experts: &[u32]) {
        let mut set: Vec<u32> = experts.to_vec();
        set.sort_unstable();
        set.dedup();

        *self.tokens.entry(layer).or_insert(0) += 1;
        let seen = self.experts.entry(layer).or_default();
        for &e in &set {
            seen.insert(e);
        }
        for &e in &set {
            *self.hits.entry((layer, e)).or_insert(0) += 1;
        }
        for (i, &a) in set.iter().enumerate() {
            for &b in &set[i + 1..] {
                *self.pairs.entry((layer, a, b)).or_insert(0) += 1;
            }
        }
    }

    /// declare an expert that exists in the model even if the trace never
    /// routed to it.
    ///
    /// a profiling corpus does not exercise everything, and an expert missing
    /// from the layout is a missing expert, not a cold one.
    pub fn declare(&mut self, layer: u32, expert: u32) {
        self.experts.entry(layer).or_default().insert(expert);
    }

    /// layers with at least one observation, in order.
    pub fn layers(&self) -> impl Iterator<Item = u32> + '_ {
        self.experts.keys().copied()
    }

    /// experts known in one layer, in index order.
    #[must_use]
    pub fn experts_in(&self, layer: u32) -> Vec<u32> {
        self.experts
            .get(&layer)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// tokens observed in one layer.
    #[must_use]
    pub fn tokens(&self, layer: u32) -> u64 {
        self.tokens.get(&layer).copied().unwrap_or(0)
    }

    /// how often one expert was routed to.
    #[must_use]
    pub fn hit_count(&self, key: ExpertKey) -> u64 {
        self.hits
            .get(&(key.layer, key.expert))
            .copied()
            .unwrap_or(0)
    }

    /// the co-activation graph, as joint probabilities.
    ///
    /// `min_weight` drops pairs too rare to be worth carrying. the graph is
    /// stored in the layout file and consulted at runtime, so leaving in
    /// hundreds of thousands of edges at a joint probability of 0.001 costs
    /// real bytes to say nothing.
    #[must_use]
    pub fn edges(&self, min_weight: f32) -> Vec<CoactivationEdge> {
        let mut out = Vec::new();
        for (&(layer, a, b), &count) in &self.pairs {
            let tokens = self.tokens(layer);
            if tokens == 0 {
                continue;
            }
            let weight = count as f32 / tokens as f32;
            if weight >= min_weight {
                out.push(CoactivationEdge {
                    layer,
                    a,
                    b,
                    weight,
                });
            }
        }
        out
    }

    /// edges for one layer, heaviest first, with a deterministic tie break.
    #[must_use]
    pub fn layer_edges(&self, layer: u32, min_weight: f32) -> Vec<CoactivationEdge> {
        let mut edges: Vec<_> = self
            .edges(min_weight)
            .into_iter()
            .filter(|e| e.layer == layer)
            .collect();
        edges.sort_by(|x, y| {
            y.weight
                .total_cmp(&x.weight)
                .then_with(|| (x.a, x.b).cmp(&(y.a, y.b)))
        });
        edges
    }
}
