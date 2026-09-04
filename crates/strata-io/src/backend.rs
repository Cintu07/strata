//! the storage interface the engine schedules against.

use crate::buffer::SlotId;
use std::io;

/// one read the engine wants performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadOp {
    /// caller's tag, handed back on completion. the engine puts the expert-layer
    /// pair or the read-plan request index here.
    pub id: u64,
    /// absolute byte offset in the file. must be alignment-aligned for direct io.
    pub offset: u64,
    /// bytes to transfer. must be alignment-aligned and fit in a slot.
    pub len: usize,
}

/// a finished read.
#[derive(Debug)]
pub struct Completion {
    /// the tag from the [`ReadOp`].
    pub id: u64,
    /// where the bytes are. release it with [`Storage::release`] once consumed.
    pub slot: SlotId,
    /// bytes actually transferred, or the error the kernel returned.
    pub result: io::Result<usize>,
}

/// how a backend is configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageConfig {
    /// how many reads may be in flight at once.
    ///
    /// this is the single most important number in the whole storage layer.
    /// consumer nvme at queue depth one is catastrophically slow on anything
    /// but a long sequential stream, and the only way to get the advertised
    /// bandwidth out of it is to keep many requests outstanding. run the
    /// `bandwidth` binary against the target device before choosing.
    pub queue_depth: usize,
    /// bytes per slot, which caps the largest single transfer.
    ///
    /// must be at least as large as the read planner's `max_request_bytes`, or
    /// a legitimately planned transfer will be refused.
    pub slot_bytes: usize,
    /// alignment for offsets, lengths and buffer addresses.
    pub alignment: usize,
    /// bypass the page cache.
    ///
    /// on by default, and the reason is a ram budget rather than speed: the
    /// page cache would hold a second copy of every expert the cache is already
    /// holding, spending the ram this whole system exists to conserve.
    pub direct: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            queue_depth: 64,
            slot_bytes: 4 << 20,
            alignment: 4096,
            direct: true,
        }
    }
}

impl StorageConfig {
    /// whether an op satisfies the alignment and size rules.
    #[must_use]
    pub fn accepts(&self, op: &ReadOp) -> bool {
        op.len <= self.slot_bytes
            && (!self.direct
                || (op.offset as usize % self.alignment == 0 && op.len % self.alignment == 0))
    }
}

/// a source of expert bytes.
///
/// the shape is submit-then-wait rather than read-and-block, because the entire
/// point of the storage design is to have many reads outstanding while compute
/// proceeds. a blocking `read_at` interface cannot express that, and adopting
/// one anywhere in the engine would quietly cap throughput at queue depth one.
pub trait Storage {
    /// how many reads may be outstanding.
    fn queue_depth(&self) -> usize;

    /// how many more reads can be accepted right now.
    fn available(&self) -> usize;

    /// queue one read.
    ///
    /// returns `Ok(None)` when every slot is busy, which is backpressure and not
    /// an error: the caller should wait for completions and try again.
    ///
    /// # Errors
    /// fails if the op violates the alignment or size rules, or if the kernel
    /// refuses the submission.
    fn submit(&mut self, op: ReadOp) -> io::Result<Option<SlotId>>;

    /// hand everything queued to the kernel.
    ///
    /// separate from [`Storage::submit`] so that a batch of reads costs one
    /// syscall rather than one each, which is most of the reason `io_uring` is
    /// worth using at all.
    ///
    /// # Errors
    /// fails if the kernel refuses the submission.
    fn flush(&mut self) -> io::Result<usize>;

    /// block until at least `min_complete` reads have finished, appending them
    /// to `out`.
    ///
    /// passing zero polls without blocking.
    ///
    /// # Errors
    /// fails if the kernel returns an error from the wait itself. a read that
    /// failed is reported inside its own [`Completion`], not here, so that one
    /// bad read does not discard the batch it arrived with.
    fn wait(&mut self, min_complete: usize, out: &mut Vec<Completion>) -> io::Result<()>;

    /// the bytes a completion delivered.
    fn bytes(&self, slot: SlotId, len: usize) -> &[u8];

    /// return a slot to the free list once its bytes have been consumed.
    fn release(&mut self, slot: SlotId);

    /// number of reads currently outstanding.
    fn in_flight(&self) -> usize;
}
