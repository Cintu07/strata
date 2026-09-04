//! both backends must agree, and the fast one must survive the things that make
//! direct io hard: alignment rules, backpressure, short reads, and being
//! dropped with work outstanding.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use strata_io::{Completion, PreadBackend, ReadOp, Storage, StorageConfig};

const ALIGN: usize = 4096;
const BLOCKS: usize = 64;

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempFile(PathBuf);

impl TempFile {
    /// a file of `BLOCKS` aligned blocks, each filled with its own block index.
    fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("strata-io-{tag}-{}-{n}.bin", std::process::id()));
        let mut data = vec![0u8; BLOCKS * ALIGN];
        for block in 0..BLOCKS {
            data[block * ALIGN..(block + 1) * ALIGN].fill(block as u8);
        }
        std::fs::write(&p, &data).unwrap();
        Self(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn config(queue_depth: usize, direct: bool) -> StorageConfig {
    StorageConfig {
        queue_depth,
        slot_bytes: 8 * ALIGN,
        alignment: ALIGN,
        direct,
    }
}

/// submit every op, draining completions whenever the queue fills, and return
/// the bytes keyed by op id. this is the shape the engine's read loop has.
fn read_all(storage: &mut dyn Storage, ops: &[ReadOp]) -> Vec<(u64, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pending: Vec<Completion> = Vec::new();
    let mut queued = 0usize;
    let mut next = 0usize;

    while next < ops.len() || queued > 0 {
        while next < ops.len() {
            match storage.submit(ops[next]).unwrap() {
                Some(_) => {
                    queued += 1;
                    next += 1;
                }
                None => break, // backpressure, go and reap
            }
        }
        storage.flush().unwrap();

        let want = if next < ops.len() { 1 } else { queued };
        storage.wait(want.min(queued).max(1), &mut pending).unwrap();

        for c in pending.drain(..) {
            let len = ops.iter().find(|o| o.id == c.id).unwrap().len;
            let bytes = c
                .result
                .as_ref()
                .map(|_| storage.bytes(c.slot, len).to_vec());
            storage.release(c.slot);
            queued -= 1;
            out.push((c.id, bytes.unwrap()));
        }
    }
    out.sort_by_key(|(id, _)| *id);
    out
}

fn scattered_ops() -> Vec<ReadOp> {
    // deliberately out of order and of differing sizes, since that is what a
    // real read plan looks like after coalescing
    [
        (0usize, 1usize),
        (17, 2),
        (3, 1),
        (40, 4),
        (9, 1),
        (60, 2),
        (25, 3),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(block, blocks))| ReadOp {
        id: i as u64,
        offset: (block * ALIGN) as u64,
        len: blocks * ALIGN,
    })
    .collect()
}

/// every byte of block `b` should be `b`, which makes a wrong offset obvious.
fn check_contents(ops: &[ReadOp], results: &[(u64, Vec<u8>)]) {
    for (id, bytes) in results {
        let op = ops.iter().find(|o| o.id == *id).unwrap();
        assert_eq!(bytes.len(), op.len, "op {id} returned the wrong length");
        for (i, chunk) in bytes.chunks(ALIGN).enumerate() {
            let expected = (op.offset as usize / ALIGN + i) as u8;
            assert!(
                chunk.iter().all(|&b| b == expected),
                "op {id} block {i} should be all {expected}, got {}",
                chunk[0]
            );
        }
    }
}

#[test]
fn pread_reads_the_right_bytes() {
    let f = TempFile::new("pread");
    let mut s = PreadBackend::open(f.path(), config(4, false)).unwrap();
    let ops = scattered_ops();
    let got = read_all(&mut s, &ops);
    assert_eq!(got.len(), ops.len());
    check_contents(&ops, &got);
}

#[test]
fn backpressure_is_reported_rather_than_blocking_or_failing() {
    let f = TempFile::new("backpressure");
    let mut s = PreadBackend::open(f.path(), config(2, false)).unwrap();

    let op = |i: u64| ReadOp {
        id: i,
        offset: i * ALIGN as u64,
        len: ALIGN,
    };
    assert!(s.submit(op(0)).unwrap().is_some());
    assert!(s.submit(op(1)).unwrap().is_some());
    assert_eq!(s.available(), 0);
    assert!(
        s.submit(op(2)).unwrap().is_none(),
        "a full queue is backpressure, not an error"
    );

    let mut done = Vec::new();
    s.wait(2, &mut done).unwrap();
    assert_eq!(done.len(), 2);
    for c in done {
        s.release(c.slot);
    }
    assert_eq!(s.available(), 2);
    assert!(
        s.submit(op(2)).unwrap().is_some(),
        "released slots become usable again"
    );
}

#[test]
fn a_read_past_the_end_of_the_file_is_an_error_not_a_panic() {
    let f = TempFile::new("eof");
    let mut s = PreadBackend::open(f.path(), config(2, false)).unwrap();
    let past = ReadOp {
        id: 0,
        offset: (BLOCKS * ALIGN) as u64,
        len: ALIGN,
    };

    s.submit(past).unwrap();
    let mut done = Vec::new();
    s.wait(1, &mut done).unwrap();
    assert_eq!(done.len(), 1);
    assert!(done[0].result.is_err(), "reading past the end should fail");
}

// ------------------------------------------------------------------- linux

#[cfg(target_os = "linux")]
mod uring {
    use super::{ALIGN, BLOCKS, TempFile, check_contents, config, read_all, scattered_ops};
    use strata_io::{PreadBackend, ReadOp, Storage, StorageConfig, UringBackend};

    /// direct io is refused by some filesystems, so a test that cannot get it
    /// should say so rather than fail and be silently ignored afterwards.
    fn open_direct(path: &std::path::Path, depth: usize) -> Option<UringBackend> {
        match UringBackend::open(path, config(depth, true)) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping: O_DIRECT unavailable here ({e})");
                None
            }
        }
    }

    #[test]
    fn io_uring_and_pread_return_identical_bytes() {
        let f = TempFile::new("parity");
        let ops = scattered_ops();

        let mut slow = PreadBackend::open(f.path(), config(4, false)).unwrap();
        let expected = read_all(&mut slow, &ops);

        let Some(mut fast) = open_direct(f.path(), 4) else {
            return;
        };
        let got = read_all(&mut fast, &ops);

        assert_eq!(got, expected, "the fast path disagreed with the reference");
        check_contents(&ops, &got);
    }

    #[test]
    fn a_deep_queue_completes_every_read_exactly_once() {
        let f = TempFile::new("deep");
        let Some(mut s) = open_direct(f.path(), 32) else {
            return;
        };

        let ops: Vec<ReadOp> = (0..BLOCKS as u64)
            .map(|b| ReadOp {
                id: b,
                offset: b * ALIGN as u64,
                len: ALIGN,
            })
            .collect();

        let got = read_all(&mut s, &ops);
        assert_eq!(got.len(), ops.len());
        let ids: Vec<u64> = got.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            (0..BLOCKS as u64).collect::<Vec<_>>(),
            "ids must be unique and complete"
        );
        check_contents(&ops, &got);
        assert_eq!(s.in_flight(), 0);
    }

    #[test]
    fn direct_io_rejects_a_misaligned_read_before_the_kernel_does() {
        let f = TempFile::new("misaligned");
        let Some(mut s) = open_direct(f.path(), 4) else {
            return;
        };

        // the kernel would return EINVAL for these. catching them here means the
        // error names the offending offset instead of a bare errno.
        let bad_offset = ReadOp {
            id: 0,
            offset: 512,
            len: ALIGN,
        };
        let bad_len = ReadOp {
            id: 1,
            offset: 0,
            len: 100,
        };
        let too_big = ReadOp {
            id: 2,
            offset: 0,
            len: 64 * ALIGN,
        };

        for op in [bad_offset, bad_len, too_big] {
            let err = s.submit(op).unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "op {} was accepted",
                op.id
            );
        }
    }

    #[test]
    fn buffered_mode_works_for_filesystems_that_refuse_direct_io() {
        let f = TempFile::new("buffered");
        // some sandboxes forbid io_uring outright, so this skips rather than
        // failing for a reason that has nothing to do with the code
        let Ok(mut s) = UringBackend::open(f.path(), config(4, false)) else {
            eprintln!("skipping: io_uring unavailable here");
            return;
        };
        assert!(!s.is_direct());

        let ops = scattered_ops();
        let got = read_all(&mut s, &ops);
        check_contents(&ops, &got);
    }

    /// dropping a backend with reads still outstanding must not let the kernel
    /// write into freed memory. the drain lives in `Drop`; this test is what
    /// exercises it, and under a sanitiser it is what would catch its absence.
    #[test]
    fn dropping_with_reads_in_flight_drains_rather_than_freeing_under_the_kernel() {
        let f = TempFile::new("dropdrain");
        let Some(mut s) = open_direct(f.path(), 16) else {
            return;
        };

        for b in 0..16u64 {
            s.submit(ReadOp {
                id: b,
                offset: b * ALIGN as u64,
                len: ALIGN,
            })
            .unwrap();
        }
        s.flush().unwrap();
        assert!(s.in_flight() > 0, "reads should still be outstanding");
        drop(s); // must block until the kernel is finished with the pool
    }

    #[test]
    fn a_short_read_is_reported_against_the_op_that_caused_it() {
        let f = TempFile::new("uringeof");
        let Some(mut s) = open_direct(f.path(), 4) else {
            return;
        };

        let good = ReadOp {
            id: 0,
            offset: 0,
            len: ALIGN,
        };
        let past = ReadOp {
            id: 1,
            offset: (BLOCKS * ALIGN) as u64,
            len: ALIGN,
        };
        s.submit(good).unwrap();
        s.submit(past).unwrap();

        let mut done = Vec::new();
        s.wait(2, &mut done).unwrap();
        assert_eq!(done.len(), 2);

        // one bad read must not discard the good one it arrived with
        let ok = done.iter().find(|c| c.id == 0).unwrap();
        let bad = done.iter().find(|c| c.id == 1).unwrap();
        assert!(ok.result.is_ok());
        assert!(bad.result.is_err());
    }

    #[test]
    fn open_best_reports_which_backend_it_got() {
        let f = TempFile::new("best");
        let cfg = StorageConfig {
            queue_depth: 4,
            slot_bytes: 8 * ALIGN,
            ..Default::default()
        };
        let (_storage, name) = strata_io::open_best(f.path(), cfg).unwrap();
        assert!(
            name == "io_uring" || name == "pread fallback",
            "unexpected backend name {name}"
        );
    }
}
