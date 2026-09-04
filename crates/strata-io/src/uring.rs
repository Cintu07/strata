//! the `io_uring` backend: many reads outstanding, direct to the device.
//!
//! # why this and not mmap, once more with the mechanism
//!
//! a page fault is a synchronous stall the scheduler cannot see coming or
//! measure afterwards. `io_uring` inverts that: reads are handed to the kernel in
//! a batch, compute continues, and completions arrive with the exact identity
//! of what landed. the prefetcher can only decide whether a weight will arrive
//! in time if it knows when reads finish, and this is the interface that tells
//! it.
//!
//! # the soundness argument
//!
//! this is the one crate in the workspace that handles raw pointers, so the
//! invariant is worth stating precisely. when a read is submitted, the kernel
//! is given a pointer into the slot pool and will write to it at some later
//! point, entirely outside rust's view. that is safe here because:
//!
//! 1. the pool is one allocation made at construction that never grows, never
//!    moves, and lives as long as the backend
//! 2. a slot is only handed to the kernel when it is on the free list, and it
//!    leaves the free list at the moment of submission
//! 3. a slot returns to the free list only through [`Storage::release`], which
//!    the caller can only reach after seeing that slot in a completion
//! 4. **the backend drains every outstanding read in `Drop` before the pool is
//!    freed**, so the kernel can never write into memory that has been returned
//!    to the allocator
//!
//! point four is the one that is easy to miss and impossible to debug. dropping
//! a ring with reads in flight is a use after free that shows up as corrupted
//! bytes in an unrelated allocation, minutes later.

use crate::backend::{Completion, ReadOp, Storage, StorageConfig};
use crate::buffer::{SlotId, SlotPool};
use io_uring::{IoUring, opcode, types};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// an `io_uring` backed reader.
pub struct UringBackend {
    file: File,
    ring: IoUring,
    pool: SlotPool,
    config: StorageConfig,
    free: Vec<SlotId>,
    /// what each busy slot is doing, indexed by slot id.
    inflight: Vec<Option<ReadOp>>,
    /// submitted to the ring but not yet handed to the kernel.
    unsubmitted: usize,
    outstanding: usize,
}

// the ring, the file and the slot pool have no useful debug output, and
// printing a buffer pool would be pages of noise, so this shows the
// configuration and the queue state instead of every field.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for UringBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UringBackend")
            .field("queue_depth", &self.config.queue_depth)
            .field("slot_bytes", &self.config.slot_bytes)
            .field("direct", &self.config.direct)
            .field("outstanding", &self.outstanding)
            .finish()
    }
}

impl UringBackend {
    /// open a file and set up a ring for it.
    ///
    /// with `config.direct` the file is opened `O_DIRECT`, which makes the
    /// kernel enforce the alignment rules rather than silently going through
    /// the page cache.
    ///
    /// # Errors
    /// fails if the file cannot be opened, if the kernel refuses to create a
    /// ring of this size, or if `O_DIRECT` is not supported by the filesystem.
    /// that last one is common and worth handling rather than panicking on:
    /// tmpfs, overlayfs and network mounts often refuse it.
    pub fn open(path: impl AsRef<Path>, config: StorageConfig) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        if config.direct {
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts.open(path.as_ref()).map_err(|e| {
            if config.direct && e.raw_os_error() == Some(libc::EINVAL) {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "O_DIRECT refused for {}. some filesystems do not support it, \
                         including tmpfs and many network mounts. set \
                         StorageConfig::direct to false to fall back to buffered reads, \
                         and expect the page cache to double count the ram budget",
                        path.as_ref().display()
                    ),
                )
            } else {
                e
            }
        })?;

        let ring = IoUring::new(config.queue_depth as u32)?;
        let pool = SlotPool::new(config.queue_depth, config.slot_bytes, config.alignment);
        let free = (0..config.queue_depth as u32).map(SlotId).rev().collect();

        Ok(Self {
            file,
            ring,
            pool,
            config,
            free,
            inflight: (0..config.queue_depth).map(|_| None).collect(),
            unsubmitted: 0,
            outstanding: 0,
        })
    }

    /// whether this backend is bypassing the page cache.
    #[must_use]
    pub const fn is_direct(&self) -> bool {
        self.config.direct
    }

    /// drain the completion queue into `out`.
    fn reap(&mut self, out: &mut Vec<Completion>) {
        let pool = &self.pool;
        let inflight = &mut self.inflight;
        let free = &mut self.free;
        let mut reaped = 0usize;

        for cqe in self.ring.completion() {
            let slot = SlotId(cqe.user_data() as u32);
            let Some(op) = inflight[slot.0 as usize].take() else {
                // a completion for a slot nobody is waiting on means the
                // bookkeeping is wrong, and continuing would hand out a slot
                // that is still being written to
                debug_assert!(false, "completion for idle slot {}", slot.0);
                continue;
            };
            reaped += 1;

            let raw = cqe.result();
            let result = if raw < 0 {
                free.push(slot); // failed reads carry no bytes, so reclaim now
                Err(io::Error::from_raw_os_error(-raw))
            } else if raw as usize != op.len {
                free.push(slot);
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("short read: asked for {} bytes, got {raw}", op.len),
                ))
            } else {
                Ok(raw as usize)
            };
            let _ = pool;
            out.push(Completion {
                id: op.id,
                slot,
                result,
            });
        }
        self.outstanding -= reaped;
    }
}

impl Storage for UringBackend {
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
                     or exceeds the {} byte slot. with O_DIRECT the kernel would \
                     reject this as EINVAL",
                    op.len, op.offset, self.config.alignment, self.config.slot_bytes
                ),
            ));
        }
        let Some(slot) = self.free.pop() else {
            return Ok(None);
        };

        let entry = opcode::Read::new(
            types::Fd(self.file.as_raw_fd()),
            self.pool.as_mut_ptr(slot),
            op.len as u32,
        )
        .offset(op.offset)
        .build()
        .user_data(u64::from(slot.0));

        // SAFETY: the buffer is a slot in a pool that outlives every operation
        // against it, the slot has just been taken off the free list so nothing
        // else refers to it, and it cannot return to the free list until its
        // completion has been reaped. see the module header.
        let pushed = unsafe { self.ring.submission().push(&entry) };

        if pushed.is_err() {
            // the submission queue is full. push what is already queued and try
            // once more before reporting backpressure.
            self.ring.submit()?;
            self.unsubmitted = 0;
            // SAFETY: as above.
            if unsafe { self.ring.submission().push(&entry) }.is_err() {
                self.free.push(slot);
                return Ok(None);
            }
        }

        self.inflight[slot.0 as usize] = Some(op);
        self.unsubmitted += 1;
        self.outstanding += 1;
        Ok(Some(slot))
    }

    fn flush(&mut self) -> io::Result<usize> {
        if self.unsubmitted == 0 {
            return Ok(0);
        }
        let n = self.ring.submit()?;
        self.unsubmitted = 0;
        Ok(n)
    }

    fn wait(&mut self, min_complete: usize, out: &mut Vec<Completion>) -> io::Result<()> {
        let want = min_complete.min(self.outstanding);
        if want > 0 {
            self.ring.submit_and_wait(want)?;
            self.unsubmitted = 0;
        } else {
            self.flush()?;
        }
        self.reap(out);
        Ok(())
    }

    fn bytes(&self, slot: SlotId, len: usize) -> &[u8] {
        self.pool.slice(slot, len)
    }

    fn release(&mut self, slot: SlotId) {
        debug_assert!(
            self.inflight[slot.0 as usize].is_none(),
            "slot {} released while a read is still using it",
            slot.0
        );
        self.free.push(slot);
    }

    fn in_flight(&self) -> usize {
        self.outstanding
    }
}

impl Drop for UringBackend {
    /// wait for every outstanding read before the pool is freed.
    ///
    /// without this, dropping a backend with reads in flight lets the kernel
    /// write into memory the allocator has already handed to someone else. it
    /// presents as corruption in an unrelated structure some time later, which
    /// is close to undebuggable.
    fn drop(&mut self) {
        let mut scratch = Vec::new();
        while self.outstanding > 0 {
            let want = self.outstanding;
            if self.ring.submit_and_wait(want).is_err() {
                // nothing useful is left to do, and leaking the pool is far
                // better than freeing memory the kernel still owns
                std::mem::forget(std::mem::replace(
                    &mut self.pool,
                    SlotPool::new(1, self.config.alignment, self.config.alignment),
                ));
                return;
            }
            self.reap(&mut scratch);
            scratch.clear();
        }
    }
}
