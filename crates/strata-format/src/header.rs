//! the fixed 128 byte file header.
//!
//! ```text
//! offset size field
//!      0    8 magic            b"STRATA\0\0"
//!      8    4 format_version   u32
//!     12    4 flags            u32
//!     16    4 n_layers         u32
//!     20    4 n_entries        u32
//!     24    8 index_off        u64, absolute
//!     32    8 index_len        u64
//!     40    4 index_crc32      u32
//!     44    4 alignment        u32, always 4096 in v1
//!     48    8 data_off         u64, absolute, alignment multiple
//!     56    8 data_len         u64
//!     64    4 model_id_len     u32
//!     68    4 n_edges          u32
//!     72    4 header_crc32     u32, over bytes [0, 72)
//!     76   52 reserved         zero
//! ```
//!
//! the model id string follows immediately at byte 128 and the whole block is
//! padded to the alignment, so a reader gets the header, the identity of the
//! model, and the position of everything else from one aligned 4kb read.

use crate::codec::{Reader, Writer};
use crate::crc32::crc32;
use crate::error::{Error, Result};
use crate::{ALIGNMENT, HEADER_LEN, MAGIC};

/// parsed file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// format version of the file on disk.
    pub format_version: u32,
    /// reserved feature bits, zero in v1.
    pub flags: u32,
    /// number of distinct layers present.
    pub n_layers: u32,
    /// number of expert entries in the index.
    pub n_entries: u32,
    /// absolute offset of the index region.
    pub index_off: u64,
    /// byte length of the index region.
    pub index_len: u64,
    /// crc32 of the index region.
    pub index_crc32: u32,
    /// alignment every expert payload starts on.
    pub alignment: u32,
    /// absolute offset of the data region.
    pub data_off: u64,
    /// byte length of the data region, padding included.
    pub data_len: u64,
    /// byte length of the model id string at offset [`HEADER_LEN`].
    pub model_id_len: u32,
    /// number of co-activation edges in the index region.
    pub n_edges: u32,
}

impl Header {
    /// serialise to exactly [`HEADER_LEN`] bytes, checksum included.
    pub(crate) fn encode(&self) -> [u8; HEADER_LEN as usize] {
        let mut w = Writer::with_capacity(HEADER_LEN as usize);
        w.bytes(&MAGIC);
        w.u32(self.format_version);
        w.u32(self.flags);
        w.u32(self.n_layers);
        w.u32(self.n_entries);
        w.u64(self.index_off);
        w.u64(self.index_len);
        w.u32(self.index_crc32);
        w.u32(self.alignment);
        w.u64(self.data_off);
        w.u64(self.data_len);
        w.u32(self.model_id_len);
        w.u32(self.n_edges);
        debug_assert_eq!(w.len(), 72);
        let checksum = crc32(w.as_slice());
        w.u32(checksum);
        w.zeros(HEADER_LEN as usize - w.len());

        let mut out = [0u8; HEADER_LEN as usize];
        out.copy_from_slice(w.as_slice());
        out
    }

    /// parse and validate the header block.
    ///
    /// # Errors
    /// fails on a wrong magic, a future format version, or a header checksum
    /// mismatch. a header that does not check out is not worth guessing at,
    /// because every offset in it is about to be used as a seek target.
    pub(crate) fn decode(buf: &[u8; HEADER_LEN as usize]) -> Result<Self> {
        let mut r = Reader::new(buf);
        let magic: [u8; 8] = r.array();
        if magic != MAGIC {
            return Err(Error::BadMagic(magic));
        }
        let header = Self {
            format_version: r.u32(),
            flags: r.u32(),
            n_layers: r.u32(),
            n_entries: r.u32(),
            index_off: r.u64(),
            index_len: r.u64(),
            index_crc32: r.u32(),
            alignment: r.u32(),
            data_off: r.u64(),
            data_len: r.u64(),
            model_id_len: r.u32(),
            n_edges: r.u32(),
        };
        let stored = r.u32();
        let computed = crc32(&buf[..72]);
        if stored != computed {
            return Err(Error::Corrupt {
                region: "header",
                expected: stored,
                found: computed,
            });
        }
        if header.format_version != crate::FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(header.format_version));
        }
        debug_assert_eq!(header.alignment, ALIGNMENT);
        Ok(header)
    }
}
