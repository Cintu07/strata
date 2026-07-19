//! the routing-only trace the m0 harness exports.
//!
//! # why this is a second format
//!
//! m0 stores traces as npz: hidden states, provenance, segment boundaries,
//! hundreds of megabytes, and numpy to open it. the rust side needs none of
//! that. it needs the order in which expert-layer pairs were touched, which for
//! a 2154 token granite capture is 800kb, small enough to live in a repository
//! so that a replay is reproducible rather than described.
//!
//! # layout, little endian
//!
//! ```text
//! 0   8  magic "STRTRACE"
//! 8   4  format version
//! 12  4  n_tokens
//! 16  4  n_layers
//! 20  4  n_experts
//! 24  4  top_k
//! 28  .. n_tokens * n_layers * top_k u16 expert indices, token major
//! ```
//!
//! expert indices rather than packed keys, so the reader rebuilds the
//! expert-layer pair itself. if the writer and the reader ever disagree about
//! what a key is they disagree in [`RouteTrace::load`], loudly, instead of
//! quietly inside a hit rate.
//!
//! written by `python -m strata_m0 export`.

use crate::types::ExpertKey;
use std::fs;
use std::io;
use std::path::Path;

const MAGIC: &[u8; 8] = b"STRTRACE";

/// the format version this build reads.
pub const VERSION: u32 = 1;

/// why a trace could not be read.
#[derive(Debug)]
pub enum TraceError {
    /// the file could not be read.
    Io(io::Error),
    /// the file is not a strata routing trace, or is a version this cannot read.
    Malformed(String),
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Malformed(m) => write!(f, "bad routing trace: {m}"),
        }
    }
}

impl std::error::Error for TraceError {}

impl From<io::Error> for TraceError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// observed routing, in the order a decoder touched it.
#[derive(Debug, Clone)]
pub struct RouteTrace {
    /// tokens captured.
    pub n_tokens: usize,
    /// transformer blocks that route.
    pub n_layers: usize,
    /// experts per layer.
    pub n_experts: usize,
    /// experts selected per token per layer.
    pub top_k: usize,
    /// every access, token major then layer then the top-k within that layer.
    keys: Vec<ExpertKey>,
}

impl RouteTrace {
    /// read a trace from disk.
    ///
    /// # Errors
    /// fails on io error, a bad magic or version, a body that disagrees with
    /// the header, or an expert index outside the declared range.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TraceError> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// parse a trace already in memory.
    ///
    /// # Errors
    /// as [`RouteTrace::load`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TraceError> {
        let bad = |m: String| TraceError::Malformed(m);

        if bytes.len() < 28 {
            return Err(bad("shorter than a header".into()));
        }
        if &bytes[0..8] != MAGIC {
            return Err(bad("wrong magic".into()));
        }
        let at =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

        let version = at(8);
        if version != VERSION {
            return Err(bad(format!(
                "version {version} is not the version {VERSION} this build reads"
            )));
        }

        let n_tokens = at(12) as usize;
        let n_layers = at(16) as usize;
        let n_experts = at(20) as usize;
        let top_k = at(24) as usize;

        let expected = n_tokens * n_layers * top_k;
        let body = &bytes[28..];
        if body.len() != expected * 2 {
            return Err(bad(format!(
                "header declares {expected} indices, body holds {}",
                body.len() / 2
            )));
        }

        let mut keys = Vec::with_capacity(expected);
        let mut cursor = 0usize;
        for _ in 0..n_tokens {
            for layer in 0..n_layers {
                for _ in 0..top_k {
                    let expert = u32::from(u16::from_le_bytes([body[cursor], body[cursor + 1]]));
                    if expert as usize >= n_experts {
                        return Err(bad(format!(
                            "expert {expert} is outside the {n_experts} the header declares"
                        )));
                    }
                    keys.push(ExpertKey::new(layer as u32, expert));
                    cursor += 2;
                }
            }
        }

        Ok(Self {
            n_tokens,
            n_layers,
            n_experts,
            top_k,
            keys,
        })
    }

    /// every access in order.
    #[must_use]
    pub fn keys(&self) -> &[ExpertKey] {
        &self.keys
    }

    /// distinct expert-layer pairs one token touches.
    ///
    /// a decoder walks every layer of every token in order, so this is the
    /// width of the scan a cache has to survive. below it, pure recency evicts
    /// every entry exactly before its next use.
    #[must_use]
    pub const fn working_set(&self) -> usize {
        self.n_layers * self.top_k
    }

    /// distinct expert-layer pairs in the whole model.
    #[must_use]
    pub const fn total_pairs(&self) -> usize {
        self.n_layers * self.n_experts
    }

    /// the experts one token routed to at one layer.
    ///
    /// returns an empty slice if either index is out of range.
    #[must_use]
    pub fn selection(&self, token: usize, layer: usize) -> &[ExpertKey] {
        if token >= self.n_tokens || layer >= self.n_layers {
            return &[];
        }
        let start = (token * self.n_layers + layer) * self.top_k;
        &self.keys[start..start + self.top_k]
    }
}
