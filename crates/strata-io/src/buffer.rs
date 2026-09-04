//! aligned buffer slots.
//!
//! direct io transfers straight between the device and user memory, with no
//! page cache in between, so the kernel imposes three alignment rules: the file
//! offset, the transfer length, and **the memory address** must all be
//! multiples of the block size. a normal `Vec<u8>` satisfies none of them, and
//! the failure mode is `EINVAL` from a read that looks perfectly reasonable.
//!
//! so buffers come from here. the pool is allocated once, never grows, and
//! never moves. that last property is not a convenience, it is what makes it
//! sound to hand the kernel a raw pointer into it and walk away: the address is
//! still valid when the completion arrives because nothing could have
//! reallocated it in the meantime.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;

/// identifies one slot in a [`SlotPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u32);

/// a fixed set of equally sized, block aligned buffers.
///
/// one allocation for the whole pool rather than one per slot, so the slots are
/// contiguous, the page tables are cheap, and there is exactly one pointer whose
/// lifetime has to be reasoned about.
#[derive(Debug)]
pub struct SlotPool {
    ptr: NonNull<u8>,
    layout: Layout,
    slot_bytes: usize,
    slots: usize,
}

// the pool owns its allocation exclusively and hands out access only through
// `&self` and `&mut self`, so it is safe to move between threads. it is
// deliberately not `Sync`-with-interior-mutability: concurrent access goes
// through the backend that owns it.
unsafe impl Send for SlotPool {}

impl SlotPool {
    /// allocate `slots` buffers of `slot_bytes` each, aligned to `alignment`.
    ///
    /// # Panics
    /// panics if the requested layout is invalid or the allocation fails. this
    /// happens at startup with a size the caller chose, and a storage backend
    /// that cannot get its buffers has nothing useful to do next.
    #[must_use]
    pub fn new(slots: usize, slot_bytes: usize, alignment: usize) -> Self {
        assert!(slots > 0, "a pool with no slots can never accept a read");
        assert!(
            slot_bytes % alignment == 0,
            "slot size {slot_bytes} must be a multiple of the {alignment} byte alignment, \
             or the second slot onwards would be misaligned"
        );
        let layout = Layout::from_size_align(slots * slot_bytes, alignment)
            .expect("slot pool layout is valid");

        // SAFETY: the layout has a non zero size, checked above by the
        // assertion that slots > 0 and by slot_bytes being a positive multiple
        // of the alignment.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).expect("slot pool allocation failed");

        Self {
            ptr,
            layout,
            slot_bytes,
            slots,
        }
    }

    /// number of slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.slots
    }

    /// whether the pool is empty, which [`SlotPool::new`] does not allow.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.slots == 0
    }

    /// size of each slot.
    #[must_use]
    pub const fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// raw pointer to a slot, for handing to the kernel.
    ///
    /// # Panics
    /// panics if `slot` is out of range.
    #[must_use]
    pub fn as_mut_ptr(&self, slot: SlotId) -> *mut u8 {
        assert!(
            (slot.0 as usize) < self.slots,
            "slot {} is out of range",
            slot.0
        );
        // SAFETY: the offset is within the single allocation, because the slot
        // index was just bounds checked and every slot is slot_bytes long.
        unsafe { self.ptr.as_ptr().add(slot.0 as usize * self.slot_bytes) }
    }

    /// the first `len` bytes of a slot.
    ///
    /// # Panics
    /// panics if `slot` is out of range or `len` exceeds the slot size. the
    /// second one matters: a silent truncation here would hand back a short
    /// expert that then fails a checksum a long way from the cause.
    #[must_use]
    pub fn slice(&self, slot: SlotId, len: usize) -> &[u8] {
        assert!(
            len <= self.slot_bytes,
            "asked for {len} bytes from a {} byte slot",
            self.slot_bytes
        );
        let ptr = self.as_mut_ptr(slot).cast_const();
        // SAFETY: the pointer is within the pool's allocation and the length was
        // bounds checked against the slot size. the bytes were zeroed at
        // allocation and are only ever written by a completed kernel read, so
        // they are always initialised.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// the whole of a slot, mutably, for a backend that fills it itself.
    ///
    /// # Panics
    /// panics if `slot` is out of range or `len` exceeds the slot size.
    pub fn slice_mut(&mut self, slot: SlotId, len: usize) -> &mut [u8] {
        assert!(
            len <= self.slot_bytes,
            "asked for {len} bytes from a {} byte slot",
            self.slot_bytes
        );
        let ptr = self.as_mut_ptr(slot);
        // SAFETY: as above, and `&mut self` guarantees no other reference into
        // the pool exists for the duration of the borrow.
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

impl Drop for SlotPool {
    fn drop(&mut self) {
        // SAFETY: ptr came from alloc_zeroed with exactly this layout and has
        // not been freed. the backend that owns the pool is responsible for
        // ensuring no kernel operation is still in flight against it, which it
        // does by draining in its own Drop before this one runs.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::{SlotId, SlotPool};

    #[test]
    fn every_slot_is_aligned() {
        let pool = SlotPool::new(8, 8192, 4096);
        for i in 0..8 {
            let addr = pool.as_mut_ptr(SlotId(i)) as usize;
            assert_eq!(addr % 4096, 0, "slot {i} landed at {addr:#x}");
        }
    }

    #[test]
    fn slots_do_not_overlap() {
        let mut pool = SlotPool::new(4, 4096, 4096);
        for i in 0..4u32 {
            pool.slice_mut(SlotId(i), 4096).fill(i as u8 + 1);
        }
        for i in 0..4u32 {
            assert!(
                pool.slice(SlotId(i), 4096)
                    .iter()
                    .all(|&b| b == i as u8 + 1)
            );
        }
    }

    #[test]
    fn a_fresh_pool_is_zeroed() {
        let pool = SlotPool::new(2, 4096, 4096);
        assert!(pool.slice(SlotId(0), 4096).iter().all(|&b| b == 0));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn an_out_of_range_slot_panics_rather_than_reading_a_neighbour() {
        let pool = SlotPool::new(2, 4096, 4096);
        let _ = pool.slice(SlotId(2), 4096);
    }

    #[test]
    #[should_panic(expected = "byte slot")]
    fn reading_past_a_slot_panics_rather_than_truncating() {
        let pool = SlotPool::new(2, 4096, 4096);
        let _ = pool.slice(SlotId(0), 8192);
    }
}
