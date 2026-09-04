//! the strata on-disk expert layout file.
//!
//! # why there is a file format here at all
//!
//! the obvious way to run a model that does not fit in ram is to mmap the
//! weights and let the kernel page them in. that is what most of the field
//! does and it is the wrong primitive for this problem:
//!
//! - a page fault is synchronous, so the faulting thread stalls with no way to
//!   overlap the read against compute
//! - the granularity is 4kb, and experts are 10 to 100mb
//! - the os page cache holds a second copy of bytes you are already caching,
//!   spending the ram you are short of
//! - eviction is the kernel's decision and the kernel has never heard of a
//!   router
//! - fault latency is invisible to your scheduler, so you cannot reason about
//!   whether a weight will arrive in time
//!
//! strata reads through an explicit async path instead, and that path needs a
//! file whose layout it can plan against. this crate is that file.
//!
//! # what the layout buys
//!
//! - every expert is contiguous and 4kb aligned, so one expert is one large
//!   sequential read rather than a scatter of random ones
//! - experts sit in **measured co-activation order**, not index order. if
//!   experts 3 and 47 of layer 12 tend to fire on the same token, they are
//!   neighbours on disk, and the read that fetches one gets the other nearly
//!   free. see the `strata-layout` crate for the ordering pass
//! - the index carries per expert offsets, sizes, checksums, precisions, and
//!   the co-activation graph itself, so the runtime can plan a whole prefill
//!   sweep without touching the data region
//! - precision is per expert, so a profile-guided pass can spend bits where the
//!   router actually goes
//!
//! # file structure
//!
//! ```text
//! [0, 4096)                  header, then the model id string, zero padded
//! [4096, 4096 + data_len)    expert payloads, each 4kb aligned, in disk order
//! [index_off, +index_len)    index: expert table, then co-activation edges
//! ```
//!
//! # example
//!
//! ```
//! use strata_format::{ExpertKey, LayoutReader, LayoutWriter, PlanOptions, Precision};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let dir = std::env::temp_dir().join("strata-format-doctest");
//! # std::fs::create_dir_all(&dir)?;
//! # let path = dir.join("demo.strata");
//! let mut w = LayoutWriter::create(&path, "demo-moe-8x1b")?;
//! w.push_expert(ExpertKey::new(0, 3), Precision::Q4, &vec![7u8; 8192])?;
//! w.push_expert(ExpertKey::new(0, 47), Precision::Q4, &vec![9u8; 8192])?;
//! w.finish()?;
//!
//! let r = LayoutReader::open(&path)?;
//! assert_eq!(r.model_id(), "demo-moe-8x1b");
//! assert_eq!(r.read_expert(ExpertKey::new(0, 3))?, vec![7u8; 8192]);
//!
//! // the two experts are adjacent, so one transfer covers both
//! let plan = r.plan_reads(&[ExpertKey::new(0, 3), ExpertKey::new(0, 47)], PlanOptions::default())?;
//! assert_eq!(plan.requests.len(), 1);
//! # std::fs::remove_file(&path)?;
//! # Ok(())
//! # }
//! ```

mod codec;
pub mod crc32;
mod error;
mod header;
mod index;
mod plan;
mod reader;
mod types;
mod writer;

pub use error::{Error, Result};
pub use header::Header;
pub use plan::{PlanOptions, ReadPlan, ReadRequest};
pub use reader::LayoutReader;
pub use types::{CoactivationEdge, ExpertEntry, ExpertKey, Precision};
pub use writer::LayoutWriter;

/// the eight magic bytes at offset zero.
pub const MAGIC: [u8; 8] = *b"STRATA\0\0";

/// on-disk format version this build reads and writes.
pub const FORMAT_VERSION: u32 = 1;

/// fixed size of the header struct, before the model id string.
pub const HEADER_LEN: u32 = 128;

/// alignment of the data region and of every expert payload inside it.
///
/// 4096 because that is the page size and the minimum unit of a direct io
/// transfer on every platform strata targets. direct io rejects unaligned
/// offsets outright, so this is a correctness constraint, not a tuning choice.
pub const ALIGNMENT: u32 = 4096;

/// bytes available for the model id string between the header and the data
/// region.
pub const MODEL_ID_CAPACITY: usize = ALIGNMENT as usize - HEADER_LEN as usize;

/// round up to the next [`ALIGNMENT`] multiple.
#[must_use]
pub const fn align_up(n: u64) -> u64 {
    let a = ALIGNMENT as u64;
    n.div_ceil(a) * a
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, ExpertKey, align_up};

    #[test]
    fn alignment_rounds_up_and_is_idempotent() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 4096);
        assert_eq!(align_up(4096), 4096);
        assert_eq!(align_up(4097), 8192);
        for n in [0u64, 1, 4095, 4096, 100_003] {
            assert_eq!(align_up(align_up(n)), align_up(n));
            assert_eq!(align_up(n) % u64::from(ALIGNMENT), 0);
        }
    }

    #[test]
    fn expert_key_packs_round_trip() {
        for k in [
            ExpertKey::new(0, 0),
            ExpertKey::new(127, 255),
            ExpertKey::new(u32::MAX, u32::MAX),
        ] {
            assert_eq!(ExpertKey::from_packed(k.packed()), k);
        }
        // distinct layers must not collide, which is the entire point of the type
        assert_ne!(
            ExpertKey::new(3, 5).packed(),
            ExpertKey::new(30, 5).packed()
        );
    }
}
