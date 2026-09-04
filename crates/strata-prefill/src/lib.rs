//! expert-centric prefill scheduling.
//!
//! # the reordering, and why it is the largest single win
//!
//! the standard prefill order is token-major: for each token, for each layer,
//! route and compute. every token independently demands the experts it routed
//! to, so across a 4096 token prefill the same expert is asked for hundreds of
//! times. when the weights are resident that costs nothing. when they are on
//! nvme it is the whole cost.
//!
//! prefill has a property decode does not: **every token is available at once**.
//! so the loops can be turned inside out.
//!
//! ```text
//! for each layer:
//!     attention over all tokens          (hot weights, always resident)
//!     route all tokens                   (token -> experts)
//!     invert                             (expert -> tokens)
//!     sort experts by disk offset        (a sweep, not a scatter)
//!     for each expert in that order:
//!         issue the read                 (deep queue, many in flight)
//!         apply it to every token that wanted it, as one batched gemm
//!     recombine in token order
//! ```
//!
//! that buys three things at once:
//!
//! - **each expert is read at most once per layer per prefill**, which is g4
//! - **reads are issued in disk order**, so the read planner can coalesce them
//!   into a small number of large transfers. measurement on a real device puts
//!   large reads at roughly a hundred times the throughput of small scattered
//!   ones, so this is not a marginal effect
//! - **the ffn becomes a batched gemm** over every token that wanted the
//!   expert, which raises arithmetic intensity from a matrix-vector product to
//!   something worth the cpu's time
//!
//! # the correctness trap
//!
//! reordering the loops reorders the additions, and floating point addition is
//! not associative. done naively, this returns logits that differ from the
//! reference in the last bits, which makes g5's diff meaningless and hides real
//! bugs in the noise.
//!
//! [`run_expert_major`] does not accumulate as contributions arrive. each one is
//! written into the top-k slot it belongs to and the slots are summed in order
//! at the end, exactly as the token-major loop sums them, so the two are **bit
//! identical** and the diff stays a real test. the price is a contribution
//! buffer of `n_tokens * top_k * d_model`, which is what
//! [`block_size_for_budget`] bounds.
//!
//! # example
//!
//! ```
//! use strata_prefill::{schedule_layer_with, LayerRouting};
//! use strata_format::ExpertKey;
//!
//! // four tokens, top-2 of a layer whose experts sit on disk in reverse order
//! let routing = LayerRouting::new(
//!     0,
//!     2,
//!     vec![3, 1, 3, 2, 1, 2, 3, 1],
//!     vec![0.5; 8],
//! );
//! let schedule = schedule_layer_with(&routing, |k| Some(u64::from(10 - k.expert) * 4096));
//!
//! // eight token-slots, but only three distinct experts, so three reads
//! assert_eq!(schedule.assignments, 8);
//! assert_eq!(schedule.reads, 3);
//! assert_eq!(schedule.reads_saved(), 5);
//!
//! // and they are ordered by where they sit on disk, not by expert index
//! let order: Vec<u32> = schedule.batches.iter().map(|b| b.key.expert).collect();
//! assert_eq!(order, vec![3, 2, 1]);
//! ```

mod execute;
mod routing;
mod schedule;

pub use execute::{Activations, CountingExpert, ExpertFn, run_expert_major, run_token_major};
pub use routing::{ExpertBatch, LayerRouting, TokenSlot};
pub use schedule::{LayerSchedule, block_size_for_budget, schedule_layer, schedule_layer_with};
