//! reading the safetensors container, without loading it.
//!
//! # the shape of the file
//!
//! ```text
//! 0   8   header length n, little endian u64
//! 8   n   json header: tensor name -> dtype, shape, data_offsets
//! 8+n ..  tensor data, offsets relative to the start of this region
//! ```
//!
//! # why this reads rather than loads
//!
//! the point of strata is models that do not fit in memory, so a converter that
//! begins by loading the model into memory would be the wrong shape from its
//! first line. granite is 2.7gb and the machine this was written on has 15.6gb;
//! olmoe is 13.8gb and does not fit at all.
//!
//! so this keeps the file open and seeks. [`SafeTensors::read_into`] copies one
//! slice of one tensor into a caller supplied buffer, which lets the converter
//! stream expert by expert with a working set of one expert rather than one
//! model.

use crate::json::{self, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// largest header this will read.
///
/// a real header is tens of kilobytes. the length is the first eight bytes of
/// an untrusted file and is used to size an allocation, so a corrupt or hostile
/// value has to be rejected before it becomes a `vec![0; header_len]`.
const MAX_HEADER: u64 = 64 << 20;

/// element type of a stored tensor.
///
/// only the widths the converter can currently move are listed. an unknown
/// dtype is an error rather than a guessed width, because guessing wrong
/// produces a file that is the right size and entirely wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// ieee single.
    F32,
    /// ieee half.
    F16,
    /// bfloat16.
    BF16,
    /// 8 bit unsigned.
    U8,
    /// 8 bit signed.
    I8,
}

impl Dtype {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "U8" => Some(Self::U8),
            "I8" => Some(Self::I8),
            _ => None,
        }
    }

    /// bytes per element.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::U8 | Self::I8 => 1,
        }
    }

    /// the strata precision code this maps to.
    #[must_use]
    pub const fn precision(self) -> Option<strata_format::Precision> {
        match self {
            Self::F32 => Some(strata_format::Precision::F32),
            Self::F16 => Some(strata_format::Precision::F16),
            Self::BF16 => Some(strata_format::Precision::BF16),
            Self::U8 | Self::I8 => None,
        }
    }
}

/// one tensor's entry in the header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// element type.
    pub dtype: Dtype,
    /// dimensions, outermost first.
    pub shape: Vec<u64>,
    /// start of this tensor within the data region.
    pub begin: u64,
    /// one past the end, within the data region.
    pub end: u64,
}

impl TensorInfo {
    /// total elements.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// bytes this tensor occupies.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.end - self.begin
    }

    /// bytes in one slice along the outermost dimension.
    ///
    /// for a stacked expert tensor of shape `[n_experts, ..]` this is one
    /// expert's share, which is the unit the converter moves.
    #[must_use]
    pub fn outer_stride(&self) -> Option<u64> {
        let outer = *self.shape.first()?;
        if outer == 0 {
            return None;
        }
        Some(self.bytes() / outer)
    }
}

/// anything that stops the converter reading a model.
#[derive(Debug)]
pub enum Error {
    /// the file could not be read.
    Io(io::Error),
    /// the header is not the json this expects.
    Header(String),
    /// a tensor was asked for that the header does not list.
    MissingTensor(String),
    /// a dtype this converter cannot move.
    UnsupportedDtype(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Header(m) => write!(f, "bad safetensors header: {m}"),
            Self::MissingTensor(n) => write!(f, "no tensor named {n}"),
            Self::UnsupportedDtype(d) => write!(f, "unsupported dtype {d}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// a safetensors file held open, with its header parsed.
#[derive(Debug)]
pub struct SafeTensors {
    file: File,
    path: PathBuf,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    /// open a file and parse its header, without reading any tensor data.
    ///
    /// # Errors
    /// fails on io error, or if the header is not valid safetensors json.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes);

        if header_len > MAX_HEADER {
            return Err(Error::Header(format!(
                "header claims {header_len} bytes, more than the {MAX_HEADER} allowed"
            )));
        }

        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        let parsed = json::parse(&header).map_err(|e| Error::Header(e.to_string()))?;

        let entries = parsed
            .entries()
            .ok_or_else(|| Error::Header("top level is not an object".into()))?;

        let mut tensors = BTreeMap::new();
        for (name, value) in entries {
            if name == "__metadata__" {
                continue;
            }
            tensors.insert(name.clone(), parse_tensor(name, value)?);
        }

        Ok(Self {
            file,
            path,
            data_start: 8 + header_len,
            tensors,
        })
    }

    /// the file this was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// every tensor in the header, sorted by name.
    #[must_use]
    pub const fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    /// look up one tensor.
    ///
    /// # Errors
    /// fails if the header does not list it.
    pub fn info(&self, name: &str) -> Result<&TensorInfo, Error> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    /// copy `len` bytes from `offset` within a tensor into `out`.
    ///
    /// # Errors
    /// fails on io error, or if the range is outside the tensor. reading past a
    /// tensor would silently return a neighbouring tensor's weights, which is
    /// the kind of bug that produces plausible garbage, so it is checked.
    pub fn read_into(&mut self, name: &str, offset: u64, out: &mut [u8]) -> Result<(), Error> {
        let info = self.info(name)?;
        let len = out.len() as u64;
        if offset + len > info.bytes() {
            return Err(Error::Header(format!(
                "{name}: asked for {len} bytes at {offset}, tensor holds {}",
                info.bytes()
            )));
        }
        let at = self.data_start + info.begin + offset;
        self.file.seek(SeekFrom::Start(at))?;
        self.file.read_exact(out)?;
        Ok(())
    }
}

fn parse_tensor(name: &str, value: &Value) -> Result<TensorInfo, Error> {
    let bad = |m: &str| Error::Header(format!("{name}: {m}"));

    let dtype_name = value
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("missing dtype"))?;
    let dtype =
        Dtype::parse(dtype_name).ok_or_else(|| Error::UnsupportedDtype(dtype_name.to_string()))?;

    let shape: Vec<u64> = value
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("missing shape"))?
        .iter()
        .map(|d| d.as_u64().ok_or_else(|| bad("shape holds a non integer")))
        .collect::<Result<_, _>>()?;

    let offsets = value
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("missing data_offsets"))?;
    if offsets.len() != 2 {
        return Err(bad("data_offsets is not a pair"));
    }
    let begin = offsets[0].as_u64().ok_or_else(|| bad("bad begin offset"))?;
    let end = offsets[1].as_u64().ok_or_else(|| bad("bad end offset"))?;
    if end < begin {
        return Err(bad("data_offsets runs backwards"));
    }

    let info = TensorInfo {
        dtype,
        shape,
        begin,
        end,
    };

    // the header states shape and byte range independently, so they can
    // disagree. a mismatch here means every later offset is wrong, and it is
    // cheaper to find out now than to write 2gb of shifted weights.
    let expected = info.elements() * dtype.width() as u64;
    if expected != info.bytes() {
        return Err(bad(&format!(
            "shape implies {expected} bytes, data_offsets span {}",
            info.bytes()
        )));
    }

    Ok(info)
}
