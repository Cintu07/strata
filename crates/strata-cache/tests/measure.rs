//! prints the policy comparison table. run with
//! `cargo test -p strata-cache --test measure -- --nocapture`.
//!
//! this is not an assertion suite, it is the thing you look at before writing
//! an assertion. thresholds in the real test suite were read off this output
//! rather than guessed, which is the only way a threshold means anything.

mod common;

use common::{domain_blocks, hot_set_with_noise, skewed_layer, uniform_sizes};
use strata_cache::baseline::{BeladyOracle, LruCache, Policy, without_admission};
use strata_cache::{CacheConfig, ExpertCache, ExpertDesc};
use strata_format::ExpertKey;

const EXPERT_BYTES: u64 = 1 << 20;

fn run(trace: &[ExpertKey], capacity_experts: u64) -> String {
    let capacity = capacity_experts * EXPERT_BYTES;
    let desc = ExpertDesc::plain(EXPERT_BYTES);

    let mut strata: ExpertCache<()> = ExpertCache::new(CacheConfig::with_capacity(capacity));
    let mut no_admission = without_admission(capacity);
    let mut lru = LruCache::new(capacity);
    for &k in trace {
        strata.access(k, desc);
        no_admission.access(k, desc);
        lru.access(k, desc);
    }
    let oracle = BeladyOracle::new(capacity).run(trace, &uniform_sizes(trace, EXPERT_BYTES));

    format!(
        "  lru {:.3}   strata-no-admission {:.3}   strata {:.3}   oracle {:.3}",
        lru.stats().hit_rate(),
        no_admission.stats().hit_rate(),
        strata.stats().hit_rate(),
        oracle.hit_rate(),
    )
}

#[test]
fn policy_comparison_table() {
    println!("\nhit rate by policy, cache size in experts\n");

    println!("hot set of 16 revisited, 4 one shot experts per round");
    for cap in [8, 16, 24, 32] {
        let t = hot_set_with_noise(16, 200, 4, 7);
        println!("  cap {cap:>3}{}", run(&t, cap));
    }

    println!("\nfour domains of 12 experts, 300 accesses per block, 24 blocks");
    for cap in [12, 18, 24, 36, 48] {
        let t = domain_blocks(4, 12, 300, 24, 11);
        println!("  cap {cap:>3}{}", run(&t, cap));
    }

    println!("\nskewed access over 64 experts in one layer");
    for cap in [8, 16, 24, 32] {
        let t = skewed_layer(64, 20_000, 13);
        println!("  cap {cap:>3}{}", run(&t, cap));
    }
    println!();
}
