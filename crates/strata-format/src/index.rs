//! the index region: where every expert is, and which experts fire together.
//!
//! ```text
//! u32                    n_entries
//! n_entries * 32 bytes   layer u32 | expert u32 | offset u64 | len u64
//!                        precision u8 | reserved u8 x3 | crc32 u32
//! u32                    n_edges
//! n_edges * 16 bytes     layer u32 | a u32 | b u32 | weight f32
//! ```
//!
//! the whole region is read in one shot at open time and checksummed before a
//! single offset in it is trusted. for a 128 layer model with 256 experts per
//! layer that is a megabyte, which is a rounding error against the ram budget
//! and buys the scheduler a complete map of the file without a scan.

use crate::codec::{Reader, Writer};
use crate::error::Result;
use crate::types::{CoactivationEdge, ExpertEntry, ExpertKey, Precision};

pub(crate) const ENTRY_LEN: usize = 32;
pub(crate) const EDGE_LEN: usize = 16;

pub(crate) fn encode(entries: &[ExpertEntry], edges: &[CoactivationEdge]) -> Vec<u8> {
    let mut w = Writer::with_capacity(8 + entries.len() * ENTRY_LEN + edges.len() * EDGE_LEN);
    w.u32(entries.len() as u32);
    for e in entries {
        w.u32(e.key.layer);
        w.u32(e.key.expert);
        w.u64(e.offset);
        w.u64(e.len);
        w.u8(e.precision.code());
        w.zeros(3);
        w.u32(e.crc32);
    }
    w.u32(edges.len() as u32);
    for edge in edges {
        w.u32(edge.layer);
        w.u32(edge.a);
        w.u32(edge.b);
        w.f32(edge.weight);
    }
    w.into_vec()
}

pub(crate) fn decode(buf: &[u8]) -> Result<(Vec<ExpertEntry>, Vec<CoactivationEdge>)> {
    let mut r = Reader::new(buf);
    let n_entries = r.u32() as usize;
    let mut entries = Vec::with_capacity(n_entries);
    for _ in 0..n_entries {
        let layer = r.u32();
        let expert = r.u32();
        let offset = r.u64();
        let len = r.u64();
        let precision = Precision::from_code(r.u8())?;
        r.skip(3);
        let crc32 = r.u32();
        entries.push(ExpertEntry {
            key: ExpertKey::new(layer, expert),
            offset,
            len,
            precision,
            crc32,
        });
    }
    let n_edges = r.u32() as usize;
    let mut edges = Vec::with_capacity(n_edges);
    for _ in 0..n_edges {
        edges.push(CoactivationEdge {
            layer: r.u32(),
            a: r.u32(),
            b: r.u32(),
            weight: r.f32(),
        });
    }
    debug_assert_eq!(r.remaining(), 0);
    Ok((entries, edges))
}

/// exact serialised size, so the writer can place the index without a trial run.
pub(crate) const fn encoded_len(n_entries: usize, n_edges: usize) -> usize {
    8 + n_entries * ENTRY_LEN + n_edges * EDGE_LEN
}
