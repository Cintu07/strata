//! the objects the format stores.

use crate::error::{Error, Result};
use std::fmt;

/// the unit of storage, caching, and prefetch in strata.
///
/// expert 5 in layer 3 and expert 5 in layer 30 are unrelated tensors that
/// happen to share an index. tracking them as one object is the mistake that
/// makes every downstream statistic meaningless, so the type system does not
/// let you name an expert without naming its layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertKey {
    /// zero based transformer block index.
    pub layer: u32,
    /// expert index within that layer's router.
    pub expert: u32,
}

impl ExpertKey {
    /// name an expert-layer pair.
    #[must_use]
    pub const fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }

    /// pack into a single u64, low bits expert, high bits layer.
    ///
    /// traces and sketches want a cheap dense integer key.
    #[must_use]
    pub const fn packed(self) -> u64 {
        ((self.layer as u64) << 32) | self.expert as u64
    }

    /// inverse of [`ExpertKey::packed`].
    #[must_use]
    pub const fn from_packed(v: u64) -> Self {
        Self {
            layer: (v >> 32) as u32,
            expert: (v & 0xFFFF_FFFF) as u32,
        }
    }
}

impl fmt::Display for ExpertKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}/E{}", self.layer, self.expert)
    }
}

/// how an expert's weights are stored on disk.
///
/// the format allows this to vary per expert so that a profile-guided pass can
/// keep frequently routed experts at higher precision and push the long tail
/// down, which buys capacity where it is never noticed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum Precision {
    /// ieee single.
    F32 = 0,
    /// ieee half.
    F16 = 1,
    /// bfloat16.
    BF16 = 2,
    /// 8 bit block quantised.
    Q8 = 3,
    /// 6 bit block quantised.
    Q6 = 4,
    /// 5 bit block quantised.
    Q5 = 5,
    /// 4 bit block quantised.
    Q4 = 6,
    /// 3 bit block quantised.
    Q3 = 7,
    /// 2 bit block quantised.
    Q2 = 8,
}

impl Precision {
    /// the byte written into the index.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// parse a byte from the index.
    ///
    /// # Errors
    /// returns [`Error::UnknownPrecision`] if the code is not one this build knows.
    pub const fn from_code(code: u8) -> Result<Self> {
        Ok(match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::BF16,
            3 => Self::Q8,
            4 => Self::Q6,
            5 => Self::Q5,
            6 => Self::Q4,
            7 => Self::Q3,
            8 => Self::Q2,
            other => return Err(Error::UnknownPrecision(other)),
        })
    }

    /// nominal bits per weight, used for capacity planning and for reporting.
    #[must_use]
    pub const fn bits_per_weight(self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::Q8 => 8.5,
            Self::Q6 => 6.5,
            Self::Q5 => 5.5,
            Self::Q4 => 4.5,
            Self::Q3 => 3.4,
            Self::Q2 => 2.6,
        }
    }

    /// whether a hit must pay a dequantisation pass before the gemm.
    #[must_use]
    pub const fn needs_dequant(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

/// where one expert lives in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertEntry {
    /// which expert-layer pair this is.
    pub key: ExpertKey,
    /// byte offset from the start of the data region. always a multiple of [`crate::ALIGNMENT`].
    pub offset: u64,
    /// payload length in bytes, before alignment padding.
    pub len: u64,
    /// storage precision of this expert.
    pub precision: Precision,
    /// crc32 of the payload bytes.
    pub crc32: u32,
}

impl ExpertEntry {
    /// absolute offset in the file.
    #[must_use]
    pub const fn file_offset(&self, data_off: u64) -> u64 {
        data_off + self.offset
    }

    /// payload length rounded up to the alignment, which is what a direct io
    /// read actually transfers.
    #[must_use]
    pub const fn padded_len(&self) -> u64 {
        crate::align_up(self.len)
    }
}

/// one edge of the measured co-activation graph.
///
/// weight is the empirical probability that `b` is routed in the same token as
/// `a`, within the layer, measured on the profiling corpus. the graph is what
/// [`strata-layout`](https://docs.rs/strata-layout) orders the file by, and what
/// the runtime consults when a miss is about to become a read anyway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoactivationEdge {
    /// the layer both experts belong to.
    pub layer: u32,
    /// first expert index.
    pub a: u32,
    /// second expert index.
    pub b: u32,
    /// measured joint probability, in `[0, 1]`.
    pub weight: f32,
}
