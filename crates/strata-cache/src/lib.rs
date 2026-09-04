//! the strata expert cache.
//!
//! # why this crate is the product
//!
//! single stream decode is bandwidth bound. the ceiling is
//!
//! ```text
//! tokens/sec = effective bandwidth / bytes that must move per token
//! ```
//!
//! for a 120b class model at four bits, expert reads are roughly 1.5gb per
//! token if nothing is cached. at a sequential 6 GB/s that is four tokens a
//! second, and consumer nvme does not deliver sequential bandwidth to a random
//! access pattern anyway. at a 70 percent hit rate the same token moves about
//! 0.45gb and lands near thirteen tokens a second, and with the read overlapped
//! against compute it stops being the binding constraint at all.
//!
//! the hit rate is not a tuning parameter of the system. it is the system.
//! everything else in strata is plumbing arranged around it.
//!
//! # the unit of caching
//!
//! [`strata_format::ExpertKey`], an expert-layer pair. expert 5 in layer 3 and
//! expert 5 in layer 30 are unrelated tensors that share an index, and merging
//! them makes every statistic downstream meaningless.
//!
//! # what is here
//!
//! - [`ExpertCache`], the shipping policy: a probationary window in front of a
//!   greedy dual size frequency main region, with tinylfu admission and
//!   optional dequantised residency for the hottest experts
//! - [`baseline`], the policies it has to beat, so that the comparison is a
//!   test rather than a claim: lru, the same policy with admission disabled,
//!   and belady's offline optimum as a ceiling
//!
//! # example
//!
//! ```
//! use strata_cache::{baseline::Policy, baseline::LruCache, CacheConfig, ExpertCache, ExpertDesc};
//! use strata_format::ExpertKey;
//!
//! // a workload that revisits a small hot set with one shot noise between
//! let hot: Vec<ExpertKey> = (0..4).map(|e| ExpertKey::new(0, e)).collect();
//! let mut trace = Vec::new();
//! for round in 0..40u32 {
//!     trace.extend_from_slice(&hot);
//!     trace.push(ExpertKey::new(0, 100 + round)); // seen once, never again
//! }
//!
//! let desc = ExpertDesc::plain(1 << 20);
//! let capacity = 4 << 20;
//!
//! let mut strata: ExpertCache<()> = ExpertCache::new(CacheConfig::with_capacity(capacity));
//! let mut lru = LruCache::new(capacity);
//! for &k in &trace {
//!     strata.access(k, desc);
//!     lru.access(k, desc);
//! }
//!
//! // admission control keeps the one shot experts from evicting the hot set
//! assert!(strata.stats().hit_rate() > lru.stats().hit_rate());
//! ```

pub mod baseline;
mod cache;
mod config;
mod sketch;
mod stats;

pub use cache::{Admission, ExpertCache, ExpertDesc, Region, Residency};
pub use config::{CacheConfig, ReadCost};
pub use stats::CacheStats;
