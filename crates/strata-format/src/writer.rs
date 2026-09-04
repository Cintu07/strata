//! building a layout file.

use crate::error::{Error, Result};
use crate::header::Header;
use crate::types::{CoactivationEdge, ExpertEntry, ExpertKey, Precision};
use crate::{
    ALIGNMENT, FORMAT_VERSION, HEADER_LEN, MODEL_ID_CAPACITY, align_up, crc32::crc32, index,
};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// writes experts to disk in the order they are pushed.
///
/// push order is disk order, and disk order is the whole point of the file, so
/// the caller is expected to have run the co-activation ordering pass first and
/// to push in the order it produced. the writer will not reorder behind your
/// back, because a silent reorder would invalidate the plan that produced it.
#[derive(Debug)]
pub struct LayoutWriter {
    out: BufWriter<File>,
    model_id: String,
    entries: Vec<ExpertEntry>,
    seen: HashSet<ExpertKey>,
    edges: Vec<CoactivationEdge>,
    cursor: u64,
}

impl LayoutWriter {
    /// create a new layout file, truncating anything already at `path`.
    ///
    /// # Errors
    /// fails if the file cannot be created, or if `model_id` does not fit in
    /// the header block.
    pub fn create(path: impl AsRef<Path>, model_id: impl Into<String>) -> Result<Self> {
        let model_id = model_id.into();
        if model_id.len() > MODEL_ID_CAPACITY {
            return Err(Error::ModelIdTooLong(model_id.len()));
        }
        let file = File::create(path)?;
        let mut out = BufWriter::new(file);
        // reserve the header block. it is rewritten with real offsets at finish.
        out.write_all(&vec![0u8; ALIGNMENT as usize])?;
        Ok(Self {
            out,
            model_id,
            entries: Vec::new(),
            seen: HashSet::new(),
            edges: Vec::new(),
            cursor: 0,
        })
    }

    /// append one expert's payload at the next aligned offset.
    ///
    /// # Errors
    /// fails on io error, or if this expert-layer pair was already written.
    pub fn push_expert(
        &mut self,
        key: ExpertKey,
        precision: Precision,
        payload: &[u8],
    ) -> Result<()> {
        if !self.seen.insert(key) {
            return Err(Error::DuplicateExpert(key));
        }
        let len = payload.len() as u64;
        self.out.write_all(payload)?;
        let padding = align_up(len) - len;
        if padding > 0 {
            self.out.write_all(&vec![0u8; padding as usize])?;
        }
        self.entries.push(ExpertEntry {
            key,
            offset: self.cursor,
            len,
            precision,
            crc32: crc32(payload),
        });
        self.cursor += align_up(len);
        Ok(())
    }

    /// attach the measured co-activation graph.
    ///
    /// edges are sorted on the way in so the reader can slice per layer without
    /// building an index of its own.
    pub fn set_coactivation(&mut self, mut edges: Vec<CoactivationEdge>) {
        edges.sort_by_key(|x| (x.layer, x.a, x.b));
        self.edges = edges;
    }

    /// number of experts written so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// whether nothing has been written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// write the index, patch the header, and fsync.
    ///
    /// # Errors
    /// fails on any io error during the final writes or the sync.
    pub fn finish(mut self) -> Result<()> {
        let data_off = u64::from(ALIGNMENT);
        let data_len = self.cursor;
        let index_off = data_off + data_len;
        let index_bytes = index::encode(&self.entries, &self.edges);
        debug_assert_eq!(
            index_bytes.len(),
            index::encoded_len(self.entries.len(), self.edges.len())
        );
        self.out.write_all(&index_bytes)?;

        let mut layers: Vec<u32> = self.entries.iter().map(|e| e.key.layer).collect();
        layers.sort_unstable();
        layers.dedup();

        let header = Header {
            format_version: FORMAT_VERSION,
            flags: 0,
            n_layers: layers.len() as u32,
            n_entries: self.entries.len() as u32,
            index_off,
            index_len: index_bytes.len() as u64,
            index_crc32: crc32(&index_bytes),
            alignment: ALIGNMENT,
            data_off,
            data_len,
            model_id_len: self.model_id.len() as u32,
            n_edges: self.edges.len() as u32,
        };

        let mut file = self
            .out
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header.encode())?;
        file.write_all(self.model_id.as_bytes())?;
        let written = HEADER_LEN as usize + self.model_id.len();
        file.write_all(&vec![0u8; ALIGNMENT as usize - written])?;
        file.sync_all()?;
        Ok(())
    }
}
