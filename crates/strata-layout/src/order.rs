//! turning a co-activation graph into a disk order.

use crate::profile::CoactivationProfile;
use std::collections::{HashMap, VecDeque};
use std::hash::BuildHasher;
use strata_format::{CoactivationEdge, ExpertKey};

/// place experts so that ones which fire together sit next to each other.
///
/// # the problem
///
/// a read that has already started is cheap to make longer. if two experts that
/// tend to be routed on the same token are adjacent on disk, the read that
/// fetches one gets the other for the cost of the bytes alone, with no second
/// request and no second round trip. so the question is how to lay a layer's
/// experts out in a line such that heavily co-activated pairs end up close.
///
/// that is minimum linear arrangement, which is np-hard, so this is a greedy
/// approximation and not an optimum.
///
/// # the algorithm
///
/// pettis and hansen's greedy chain merging, the same method used to lay out
/// procedures in a binary from a call graph. it is a good fit because the shape
/// of the problem is identical: a weighted undirected graph, a linear output,
/// and a cost that falls off with distance.
///
/// every expert starts as a chain of one. edges are taken heaviest first, and
/// an edge joins two chains if both of its endpoints are currently at the end
/// of their chain, flipping either chain as needed. an edge whose endpoints are
/// buried inside a chain is skipped, because honouring it would break bonds
/// that were already stronger. leftover chains are concatenated at the end.
///
/// # determinism
///
/// edges are sorted by weight and then by index, and leftover chains are
/// emitted in a fixed order, so the same profile always produces the same
/// layout. a layout that shuffles between runs makes every before and after
/// measurement meaningless.
#[must_use]
pub fn order_layer(experts: &[u32], edges: &[CoactivationEdge]) -> Vec<u32> {
    if experts.len() <= 1 {
        return experts.to_vec();
    }

    // chain id -> the chain, and expert -> which chain it is in
    let mut chains: HashMap<usize, VecDeque<u32>> = HashMap::new();
    let mut chain_of: HashMap<u32, usize> = HashMap::new();
    for (i, &e) in experts.iter().enumerate() {
        chains.insert(i, VecDeque::from(vec![e]));
        chain_of.insert(e, i);
    }

    let mut sorted: Vec<&CoactivationEdge> = edges.iter().collect();
    sorted.sort_by(|x, y| {
        y.weight
            .total_cmp(&x.weight)
            .then_with(|| (x.a, x.b).cmp(&(y.a, y.b)))
    });

    for edge in sorted {
        let (Some(&ca), Some(&cb)) = (chain_of.get(&edge.a), chain_of.get(&edge.b)) else {
            continue;
        };
        if ca == cb {
            continue;
        }
        // `a` has to end up at the tail of its chain and `b` at the head of
        // its own, so the join puts them next to each other. an endpoint buried
        // in the middle cannot be moved without breaking a heavier bond.
        //
        // written with `let ... else` rather than `expect` so that a bug in the
        // bookkeeping degrades into a slightly worse layout rather than a panic
        // in the middle of a model conversion.
        let Some(head) = chains.get_mut(&ca) else {
            continue;
        };
        if !orient_tail(head, edge.a) {
            continue;
        }
        let Some(next) = chains.get_mut(&cb) else {
            continue;
        };
        if !orient_head(next, edge.b) {
            continue;
        }
        let Some(tail) = chains.remove(&cb) else {
            continue;
        };
        for &e in &tail {
            chain_of.insert(e, ca);
        }
        if let Some(head) = chains.get_mut(&ca) {
            head.extend(tail);
        }
    }

    // emit chains in order of their first expert, so the result is stable
    let mut remaining: Vec<VecDeque<u32>> = chains.into_values().collect();
    remaining.sort_by_key(|c| c.front().copied().unwrap_or(u32::MAX));
    remaining.into_iter().flatten().collect()
}

/// make `e` the last element of the chain, reversing if it is first.
/// returns false if `e` is buried in the middle.
fn orient_tail(chain: &mut VecDeque<u32>, e: u32) -> bool {
    if chain.back() == Some(&e) {
        return true;
    }
    if chain.front() == Some(&e) {
        let mut v: Vec<u32> = chain.drain(..).collect();
        v.reverse();
        chain.extend(v);
        return true;
    }
    false
}

/// make `e` the first element of the chain, reversing if it is last.
fn orient_head(chain: &mut VecDeque<u32>, e: u32) -> bool {
    if chain.front() == Some(&e) {
        return true;
    }
    if chain.back() == Some(&e) {
        let mut v: Vec<u32> = chain.drain(..).collect();
        v.reverse();
        chain.extend(v);
        return true;
    }
    false
}

/// the full disk order for a model, layer by layer.
///
/// layers are kept together and in order because a token walks layers in
/// sequence: grouping by layer is what makes a prefill sweep move forward
/// through the file instead of jumping back and forth across it.
#[must_use]
pub fn plan_layout(profile: &CoactivationProfile, min_weight: f32) -> Vec<ExpertKey> {
    let mut out = Vec::new();
    for layer in profile.layers() {
        let experts = profile.experts_in(layer);
        let edges = profile.layer_edges(layer, min_weight);
        for e in order_layer(&experts, &edges) {
            out.push(ExpertKey::new(layer, e));
        }
    }
    out
}

/// how much of the co-activation weight a given order actually captures.
///
/// for each edge, if the two experts land within `window_bytes` of each other,
/// a read that bridges the gap picks up both, so that edge's weight counts as
/// captured. the result is the fraction of total weight captured, which is
/// directly the fraction of would-be second reads the layout converts into
/// bytes on a read that was happening anyway.
///
/// `window_bytes` should be the same figure as
/// [`strata_format::PlanOptions::max_gap_bytes`], because that is the gap the
/// planner will actually bridge. measuring against a window the planner would
/// never use produces a number that flatters the layout and predicts nothing.
#[must_use]
pub fn capture_ratio<S: BuildHasher>(
    order: &[u32],
    sizes: &HashMap<u32, u64, S>,
    edges: &[CoactivationEdge],
    window_bytes: u64,
) -> f64 {
    let mut offset = HashMap::with_capacity(order.len());
    let mut at = 0u64;
    for &e in order {
        offset.insert(e, at);
        at += strata_format::align_up(sizes.get(&e).copied().unwrap_or(0));
    }

    let mut total = 0.0f64;
    let mut captured = 0.0f64;
    for edge in edges {
        let (Some(&pa), Some(&pb)) = (offset.get(&edge.a), offset.get(&edge.b)) else {
            continue;
        };
        let w = f64::from(edge.weight);
        total += w;

        // the gap is between the end of the earlier expert and the start of the
        // later one, which is exactly what the read planner has to bridge
        let (lo, hi) = if pa <= pb {
            (edge.a, edge.b)
        } else {
            (edge.b, edge.a)
        };
        let lo_end = offset[&lo] + strata_format::align_up(sizes.get(&lo).copied().unwrap_or(0));
        let gap = offset[&hi].saturating_sub(lo_end);
        if gap <= window_bytes {
            captured += w;
        }
    }
    if total == 0.0 {
        return 0.0;
    }
    captured / total
}
