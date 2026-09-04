//! errors produced while reading or writing a layout file.

use std::fmt;

/// what went wrong.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// underlying io failure.
    Io(std::io::Error),
    /// the first eight bytes are not a strata layout file.
    BadMagic([u8; 8]),
    /// the file was written by a format version this build does not understand.
    UnsupportedVersion(u32),
    /// a checksummed region did not match its recorded checksum.
    Corrupt {
        /// which region failed: `"header"`, `"index"`, or `"expert"`.
        region: &'static str,
        /// checksum recorded in the file.
        expected: u32,
        /// checksum computed over the bytes actually read.
        found: u32,
    },
    /// the file is shorter than its own header says it should be.
    Truncated {
        /// byte offset the header pointed at.
        wanted: u64,
        /// bytes actually present.
        have: u64,
    },
    /// an expert-layer pair was written twice.
    DuplicateExpert(crate::ExpertKey),
    /// the requested expert-layer pair is not in this file.
    MissingExpert(crate::ExpertKey),
    /// the model identifier does not fit in the reserved header block.
    ModelIdTooLong(usize),
    /// a byte code in the index does not name a known precision.
    UnknownPrecision(u8),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::BadMagic(m) => write!(f, "not a strata layout file, magic was {m:?}"),
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "layout format version {v} is newer than this build supports"
                )
            }
            Self::Corrupt {
                region,
                expected,
                found,
            } => write!(
                f,
                "{region} checksum mismatch, expected {expected:#010x} found {found:#010x}"
            ),
            Self::Truncated { wanted, have } => {
                write!(
                    f,
                    "file truncated, header points at byte {wanted} but file is {have} bytes"
                )
            }
            Self::DuplicateExpert(k) => write!(f, "expert {k} written twice"),
            Self::MissingExpert(k) => write!(f, "expert {k} is not in this file"),
            Self::ModelIdTooLong(n) => {
                write!(
                    f,
                    "model id is {n} bytes, the header block holds at most {}",
                    crate::MODEL_ID_CAPACITY
                )
            }
            Self::UnknownPrecision(c) => write!(f, "unknown precision code {c}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
