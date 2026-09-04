//! the portable backend: positional reads, one at a time.
//!
//! this exists for two reasons and neither of them is speed.
//!
//! it is the **reference**. `io_uring` is a lot of machinery to get wrong
//! quietly, so every test that checks bytes runs against both backends and
//! compares. a difference is a bug in the interesting one.
//!
//! it is the **fallback**. development happens on windows and macos, where
//! there is no `io_uring`, and an engine that cannot run at all off linux is an
//! engine nobody contributes to.
//!
//! it serialises: a submit performs the read there and then, so the effective
//! queue depth is one no matter what the config says. on a consumer nvme that
//! is the slow path the whole storage design exists to escape, so do not
//! benchmark against it and conclude anything about the device.

use crate::backend::{Completion, ReadOp, Storage, StorageConfig};
use crate::buffer::{SlotId, SlotPool};
use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// positional reads into a slot pool.
#[derive(Debug)]
pub struct PreadBackend {
    file: File,
    pool: SlotPool,
    config: StorageConfig,
    free: Vec<SlotId>,
    done: VecDeque<Completion>,
}

impl PreadBackend {
    /// open a file for reading through this backend.
    ///
    /// `config.direct` is ignored here: portable positional reads go through
    /// the page cache, and pretending otherwise would make the ram accounting
    /// wrong on exactly the platform where it is hardest to check.
    ///
    /// # Errors
    /// fails if the file cannot be opened.
    pub fn open(path: impl AsRef<Path>, config: StorageConfig) -> io::Result<Self> {
        let file = File::open(path)?;
        let pool = SlotPool::new(config.queue_depth, config.slot_bytes, config.alignment);
        let free = (0..config.queue_depth as u32).map(SlotId).rev().collect();
        Ok(Self {
            file,
            pool,
            config,
            free,
            done: VecDeque::new(),
        })
    }
}

impl Storage for PreadBackend {
    fn queue_depth(&self) -> usize {
        self.config.queue_depth
    }

    fn available(&self) -> usize {
        self.free.len()
    }

    fn submit(&mut self, op: ReadOp) -> io::Result<Option<SlotId>> {
        if !self.config.accepts(&op) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "read of {} bytes at offset {} does not satisfy alignment {} \
                     or exceeds the {} byte slot",
                    op.len, op.offset, self.config.alignment, self.config.slot_bytes
                ),
            ));
        }
        let Some(slot) = self.free.pop() else {
            return Ok(None);
        };

        let result = read_exact_at(&self.file, self.pool.slice_mut(slot, op.len), op.offset)
            .map(|()| op.len);
        self.done.push_back(Completion {
            id: op.id,
            slot,
            result,
        });
        Ok(Some(slot))
    }

    fn flush(&mut self) -> io::Result<usize> {
        Ok(0) // nothing is deferred, so there is nothing to push
    }

    fn wait(&mut self, min_complete: usize, out: &mut Vec<Completion>) -> io::Result<()> {
        if min_complete > self.done.len() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "asked to wait for {min_complete} completions with only {} outstanding. \
                     this backend performs reads on submit, so waiting for more than have \
                     been submitted would deadlock",
                    self.done.len()
                ),
            ));
        }
        out.extend(self.done.drain(..));
        Ok(())
    }

    fn bytes(&self, slot: SlotId, len: usize) -> &[u8] {
        self.pool.slice(slot, len)
    }

    fn release(&mut self, slot: SlotId) {
        self.free.push(slot);
    }

    fn in_flight(&self) -> usize {
        self.done.len()
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
        }
        done += n;
    }
    Ok(())
}
