//! profile-guided expert placement.
//!
//! # the idea in one paragraph
//!
//! experts are written to disk in the order they tend to be used together, not
//! in index order. if experts 3 and 47 of layer 12 fire on the same token more
//! often than not, they are neighbours in the file, and the read that fetches
//! one gets the other for the price of the bytes, with no second request and no
//! second round trip. on a device where a request costs a hundred microseconds
//! and streams at five gigabytes a second, that round trip is worth about six
//! hundred kilobytes of bytes, so converting a request into bytes is a good
//! trade far more often than it looks.
//!
//! this is a profile-guided layout pass. it runs once per model, against a
//! trace from a profiling corpus, and its output is the order experts are
//! written in by [`strata_format::LayoutWriter`].
//!
//! # what is here
//!
//! - [`CoactivationProfile`], which counts what fired together
//! - [`order_layer`] and [`plan_layout`], which turn those counts into a disk
//!   order using greedy chain merging
//! - [`capture_ratio`], which says how much of the co-activation weight an
//!   order actually captures, so the pass can be measured rather than assumed
//!
//! # example
//!
//! ```
//! use std::collections::HashMap;
//! use strata_layout::{capture_ratio, order_layer, CoactivationProfile};
//!
//! // experts 0 and 9 always fire together, and so do 1 and 8, but their
//! // indices put them at opposite ends of the file
//! let mut profile = CoactivationProfile::new();
//! for _ in 0..100 {
//!     profile.observe(0, &[0, 9]);
//!     profile.observe(0, &[1, 8]);
//! }
//! for e in 0..10 {
//!     profile.declare(0, e);
//! }
//!
//! let edges = profile.layer_edges(0, 0.0);
//! let sizes: HashMap<u32, u64> = (0..10).map(|e| (e, 1 << 20)).collect();
//! let window = 0; // strictly adjacent only
//!
//! let by_index: Vec<u32> = (0..10).collect();
//! let by_coactivation = order_layer(&profile.experts_in(0), &edges);
//!
//! assert!(
//!     capture_ratio(&by_coactivation, &sizes, &edges, window)
//!         > capture_ratio(&by_index, &sizes, &edges, window)
//! );
//! ```

mod order;
mod profile;

pub use order::{capture_ratio, order_layer, plan_layout};
pub use profile::CoactivationProfile;
