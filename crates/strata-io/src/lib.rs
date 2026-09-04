//! the storage tier.
//!
//! strata treats nvme as tier 2 of the memory hierarchy rather than as a place
//! the model is loaded from, which means reads happen on the critical path of
//! every token and the engine has to be able to overlap them against compute.
//! that requirement rules out the obvious implementation, and this crate is
//! what replaces it.
//!
//! # what is wrong with mmap
//!
//! - a page fault is synchronous, so the faulting thread stalls with no way to
//!   overlap
//! - the granularity is 4kb, and experts are 10 to 100mb
//! - the page cache holds a second copy of bytes the expert cache is already
//!   holding, spending the ram the system is short of
//! - eviction becomes the kernel's decision, and the kernel has never heard of
//!   a router
//! - fault latency is invisible to the scheduler, so it cannot reason about
//!   whether a weight will arrive in time
//!
//! # what is here
//!
//! [`Storage`] is a submit-then-wait interface, deliberately. a blocking
//! `read_at` cannot express many outstanding reads, and adopting one anywhere
//! in the engine would cap throughput at queue depth one, which on consumer
//! nvme is a factor of fifty.
//!
//! - [`UringBackend`] on linux: `io_uring`, `O_DIRECT`, deep queues
//! - [`PreadBackend`] everywhere: positional reads, one at a time. the
//!   reference the fast path is diffed against, and the reason development
//!   works on windows and macos
//!
//! # measure before tuning
//!
//! the `bandwidth` binary reports achieved throughput against queue depth and
//! request size on a real device. the prd's m2 gate is 80 percent of device
//! spec on sequential reads, and its stated risk is that consumer nvme random
//! reads at shallow queue depth are far worse than the sheet claims. both are
//! questions about hardware, so run it rather than guessing:
//!
//! ```text
//! cargo run --release -p strata-io --bin bandwidth -- /path/on/the/target/device
//! ```

// the one crate in the workspace that is allowed raw pointers. handing the
// kernel a buffer address and continuing is what io_uring is, and there is no
// safe wrapper that does not reintroduce the copy this design exists to avoid.
// every unsafe block below carries a SAFETY comment, and the invariant they all
// depend on is stated in the `uring` module header.
#![allow(unsafe_code)]

mod backend;
mod buffer;
mod pread;

pub use backend::{Completion, ReadOp, Storage, StorageConfig};
pub use buffer::{SlotId, SlotPool};
pub use pread::PreadBackend;

#[cfg(target_os = "linux")]
mod uring;
#[cfg(target_os = "linux")]
pub use uring::UringBackend;

/// open the best backend this platform offers.
///
/// on linux this is `io_uring`, falling back to positional reads if the ring
/// cannot be created or the filesystem refuses `O_DIRECT`. the fallback is
/// reported rather than hidden, because it is a large performance difference
/// and silently taking it would make a benchmark meaningless.
///
/// # Errors
/// fails if the file cannot be opened by any backend.
#[cfg(target_os = "linux")]
pub fn open_best(
    path: impl AsRef<std::path::Path>,
    config: StorageConfig,
) -> std::io::Result<(Box<dyn Storage>, &'static str)> {
    if let Ok(b) = UringBackend::open(path.as_ref(), config) {
        return Ok((Box::new(b), "io_uring"));
    }
    let b = PreadBackend::open(path, config)?;
    Ok((Box::new(b), "pread fallback"))
}

/// open the best backend this platform offers.
///
/// # Errors
/// fails if the file cannot be opened.
#[cfg(not(target_os = "linux"))]
pub fn open_best(
    path: impl AsRef<std::path::Path>,
    config: StorageConfig,
) -> std::io::Result<(Box<dyn Storage>, &'static str)> {
    let b = PreadBackend::open(path, config)?;
    Ok((Box::new(b), "pread"))
}
