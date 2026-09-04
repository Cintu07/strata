//! reading a layout file.
//!
//! reads are positional, never seek plus read. a shared cursor is a false
//! dependency between io threads, and the whole storage design is built around
//! keeping many expert reads in flight at once.

use crate::error::{Error, Result};
use crate::header::Header;
use crate::index;
use crate::plan::{PlanOptions, ReadPlan};
use crate::types::{CoactivationEdge, ExpertEntry, ExpertKey};
use crate::{HEADER_LEN, crc32::crc32};
use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// an open layout file.
#[derive(Debug)]
pub struct LayoutReader {
    file: File,
    header: Header,
    model_id: String,
    /// sorted by `offset`, which is also the order the experts appear on disk.
    entries: Vec<ExpertEntry>,
    by_key: HashMap<ExpertKey, usize>,
    edges: Vec<CoactivationEdge>,
    layer_edges: HashMap<u32, Range<usize>>,
}

impl LayoutReader {
    /// open and fully validate a layout file.
    ///
    /// the header and the index are checksummed here, once, so that every later
    /// offset can be used without a second thought. expert payloads are checked
    /// on read instead, because checking sixty gigabytes at open time would
    /// defeat the purpose of the file.
    ///
    /// # Errors
    /// fails on io error, bad magic, unsupported version, truncation, or a
    /// header or index checksum mismatch.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();

        let mut head = [0u8; HEADER_LEN as usize];
        read_exact_at(&file, &mut head, 0)?;
        let header = Header::decode(&head)?;

        let end = header.index_off + header.index_len;
        if end > file_len {
            return Err(Error::Truncated {
                wanted: end,
                have: file_len,
            });
        }

        let mut id = vec![0u8; header.model_id_len as usize];
        read_exact_at(&file, &mut id, u64::from(HEADER_LEN))?;
        let model_id = String::from_utf8_lossy(&id).into_owned();

        let mut index_bytes = vec![0u8; header.index_len as usize];
        read_exact_at(&file, &mut index_bytes, header.index_off)?;
        let found = crc32(&index_bytes);
        if found != header.index_crc32 {
            return Err(Error::Corrupt {
                region: "index",
                expected: header.index_crc32,
                found,
            });
        }

        let (entries, edges) = index::decode(&index_bytes)?;
        let by_key = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.key, i))
            .collect();

        let mut layer_edges: HashMap<u32, Range<usize>> = HashMap::new();
        for (i, e) in edges.iter().enumerate() {
            layer_edges
                .entry(e.layer)
                .and_modify(|r| r.end = i + 1)
                .or_insert(i..i + 1);
        }

        Ok(Self {
            file,
            header,
            model_id,
            entries,
            by_key,
            edges,
            layer_edges,
        })
    }

    /// the model this file was built from.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// the parsed header.
    #[must_use]
    pub const fn header(&self) -> &Header {
        &self.header
    }

    /// every expert in the file, in disk order.
    #[must_use]
    pub fn entries(&self) -> &[ExpertEntry] {
        &self.entries
    }

    /// look up one expert's index entry.
    #[must_use]
    pub fn entry(&self, key: ExpertKey) -> Option<&ExpertEntry> {
        self.by_key.get(&key).map(|&i| &self.entries[i])
    }

    /// the co-activation edges measured for one layer, sorted by `(a, b)`.
    #[must_use]
    pub fn coactivation(&self, layer: u32) -> &[CoactivationEdge] {
        self.layer_edges
            .get(&layer)
            .map_or(&[], |r| &self.edges[r.clone()])
    }

    /// total bytes of expert payload, padding excluded.
    #[must_use]
    pub fn payload_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.len).sum()
    }

    /// read one expert and verify its checksum.
    ///
    /// # Errors
    /// fails if the expert is not present, on io error, or if the payload does
    /// not match the checksum recorded when it was written.
    pub fn read_expert(&self, key: ExpertKey) -> Result<Vec<u8>> {
        let entry = *self.entry(key).ok_or(Error::MissingExpert(key))?;
        let mut buf = vec![0u8; entry.len as usize];
        read_exact_at(
            &self.file,
            &mut buf,
            entry.file_offset(self.header.data_off),
        )?;
        let found = crc32(&buf);
        if found != entry.crc32 {
            return Err(Error::Corrupt {
                region: "expert",
                expected: entry.crc32,
                found,
            });
        }
        Ok(buf)
    }

    /// turn a set of wanted experts into a small number of large sequential
    /// transfers, and report what the coalescing bought.
    ///
    /// this is the call the prefill sweep and the prefetcher both go through.
    /// see [`crate::plan`] for the reasoning behind the defaults.
    ///
    /// # Errors
    /// fails if any requested expert is not in this file.
    pub fn plan_reads(&self, keys: &[ExpertKey], opts: PlanOptions) -> Result<ReadPlan> {
        let mut idx = Vec::with_capacity(keys.len());
        for &k in keys {
            idx.push(*self.by_key.get(&k).ok_or(Error::MissingExpert(k))?);
        }
        Ok(ReadPlan::build(
            &self.entries,
            &mut idx,
            self.header.data_off,
            opts,
        ))
    }

    /// execute one planned transfer and hand back the raw bytes.
    ///
    /// the caller slices individual experts out of it using
    /// [`crate::plan::ReadRequest::slice_of`]. payload checksums are not
    /// verified here, because a coalesced transfer legitimately contains bytes
    /// nobody asked for; verify the experts you actually consume.
    ///
    /// # Errors
    /// fails on io error or short read.
    pub fn execute(&self, req: &crate::plan::ReadRequest) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; req.len as usize];
        read_exact_at(&self.file, &mut buf, req.file_offset)?;
        Ok(buf)
    }
}

/// positional read that does not disturb any file cursor.
#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    file.read_exact_at(buf, offset)
}

/// windows has no `read_exact_at`, only `seek_read`, so the short read loop is
/// written out here. it is the same contract either way.
#[cfg(windows)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read at end of layout file",
            ));
        }
        done += n;
    }
    Ok(())
}
