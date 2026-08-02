//! synthetic access traces with a known shape, used to check that a policy
//! responds to that shape the way the design says it should.
//!
//! these are not measurements. they were once described as having "the
//! structure real router traces are claimed to have", and m0 has since shown
//! that claim was incomplete: none of these generators produce the cyclic scan
//! over one token's working set that a real decoder produces, and that scan is
//! what takes lru to exactly zero. read `tests/replay.rs` for the real trace.
//! these workloads isolate one property each, which is what they are for.

#![allow(dead_code)] // each test binary uses a different subset of these

use std::collections::HashMap;
use strata_format::ExpertKey;

/// small deterministic generator, so a failing assertion is reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// index into `n` items with a zipf-like skew: repeated minimum of `k`
    /// draws concentrates mass on the low indices.
    pub fn skewed(&mut self, n: u64, k: u32) -> u64 {
        (0..k).map(|_| self.below(n)).min().unwrap_or(0)
    }
}

/// a hot set that is revisited, with one shot experts mixed in.
///
/// the one shot experts are the thing admission control exists to survive: in a
/// real prefill they are the long tail the router touches once and never again.
pub fn hot_set_with_noise(
    hot: u32,
    rounds: u32,
    noise_per_round: u32,
    seed: u64,
) -> Vec<ExpertKey> {
    let mut rng = Rng::new(seed);
    let mut trace = Vec::new();
    let mut next_noise = 1000u32;
    for _ in 0..rounds {
        for _ in 0..hot {
            trace.push(ExpertKey::new(0, rng.below(u64::from(hot)) as u32));
        }
        for _ in 0..noise_per_round {
            trace.push(ExpertKey::new(0, next_noise));
            next_noise += 1;
        }
    }
    trace
}

/// several domains, each with its own hot set, visited in blocks.
///
/// this is the topic switch case: a coding conversation hits a stable subset,
/// a digression hits a different one, and then the conversation comes back.
pub fn domain_blocks(
    domains: u32,
    experts_per_domain: u32,
    block_len: u32,
    blocks: u32,
    seed: u64,
) -> Vec<ExpertKey> {
    let mut rng = Rng::new(seed);
    let mut trace = Vec::new();
    for b in 0..blocks {
        let domain = b % domains;
        let base = domain * experts_per_domain;
        for _ in 0..block_len {
            let e = base + rng.below(u64::from(experts_per_domain)) as u32;
            trace.push(ExpertKey::new(0, e));
        }
    }
    trace
}

/// skewed access over one layer, which is the load imbalance every router
/// paper reports.
pub fn skewed_layer(n_experts: u32, len: usize, seed: u64) -> Vec<ExpertKey> {
    let mut rng = Rng::new(seed);
    (0..len)
        .map(|_| ExpertKey::new(0, rng.skewed(u64::from(n_experts), 3) as u32))
        .collect()
}

/// uniform sizes, which is the case where the belady oracle is exactly optimal.
pub fn uniform_sizes(trace: &[ExpertKey], bytes: u64) -> HashMap<ExpertKey, u64> {
    trace.iter().map(|&k| (k, bytes)).collect()
}
