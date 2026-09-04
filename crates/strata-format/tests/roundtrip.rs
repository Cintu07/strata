//! the file survives a write and read cycle, and refuses to hand back bytes it
//! cannot vouch for.

mod common;

use common::{TempFile, payload};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use strata_format::{
    ALIGNMENT, CoactivationEdge, Error, ExpertKey, HEADER_LEN, LayoutReader, LayoutWriter,
    MODEL_ID_CAPACITY, Precision,
};

/// varied sizes on purpose: one byte, exactly aligned, and one byte over,
/// because the padding arithmetic is where an off by one would hide.
fn sample() -> Vec<(ExpertKey, Precision, Vec<u8>)> {
    vec![
        (ExpertKey::new(0, 3), Precision::Q4, payload(1, 1)),
        (ExpertKey::new(0, 47), Precision::Q4, payload(2, 4096)),
        (ExpertKey::new(0, 12), Precision::Q8, payload(3, 4097)),
        (ExpertKey::new(1, 3), Precision::Q2, payload(4, 12_345)),
        (ExpertKey::new(31, 255), Precision::BF16, payload(5, 65_536)),
    ]
}

fn write_sample(path: &std::path::Path) -> Vec<(ExpertKey, Precision, Vec<u8>)> {
    let items = sample();
    let mut w = LayoutWriter::create(path, "test-moe-8x1b").unwrap();
    for (k, p, bytes) in &items {
        w.push_expert(*k, *p, bytes).unwrap();
    }
    w.set_coactivation(vec![
        CoactivationEdge {
            layer: 0,
            a: 3,
            b: 47,
            weight: 0.81,
        },
        CoactivationEdge {
            layer: 1,
            a: 3,
            b: 9,
            weight: 0.42,
        },
        CoactivationEdge {
            layer: 0,
            a: 12,
            b: 3,
            weight: 0.15,
        },
    ]);
    w.finish().unwrap();
    items
}

#[test]
fn payloads_survive_the_round_trip() {
    let f = TempFile::new("roundtrip");
    let items = write_sample(f.path());
    let r = LayoutReader::open(f.path()).unwrap();

    assert_eq!(r.model_id(), "test-moe-8x1b");
    assert_eq!(r.header().n_entries, 5);
    assert_eq!(
        r.header().n_layers,
        3,
        "layers 0, 1 and 31 are three distinct layers"
    );
    assert_eq!(
        r.payload_bytes(),
        items.iter().map(|(_, _, b)| b.len() as u64).sum::<u64>()
    );

    for (k, p, bytes) in &items {
        assert_eq!(
            &r.read_expert(*k).unwrap(),
            bytes,
            "payload mismatch for {k}"
        );
        assert_eq!(r.entry(*k).unwrap().precision, *p);
    }
}

#[test]
fn every_expert_starts_on_an_alignment_boundary() {
    let f = TempFile::new("aligned");
    write_sample(f.path());
    let r = LayoutReader::open(f.path()).unwrap();

    assert_eq!(r.header().data_off % u64::from(ALIGNMENT), 0);
    for e in r.entries() {
        assert_eq!(e.offset % u64::from(ALIGNMENT), 0, "{} is unaligned", e.key);
        assert_eq!(e.file_offset(r.header().data_off) % u64::from(ALIGNMENT), 0);
    }
}

#[test]
fn entries_are_in_disk_order_and_do_not_overlap() {
    let f = TempFile::new("order");
    write_sample(f.path());
    let r = LayoutReader::open(f.path()).unwrap();

    for pair in r.entries().windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(a.offset < b.offset, "index must be sorted by offset");
        assert!(
            a.offset + a.padded_len() <= b.offset,
            "{} overlaps {}",
            a.key,
            b.key
        );
    }
}

#[test]
fn coactivation_edges_are_sliced_per_layer_and_sorted() {
    let f = TempFile::new("coact");
    write_sample(f.path());
    let r = LayoutReader::open(f.path()).unwrap();

    let l0 = r.coactivation(0);
    assert_eq!(l0.len(), 2);
    assert_eq!((l0[0].a, l0[0].b), (3, 47));
    assert!((l0[0].weight - 0.81).abs() < 1e-6);
    assert_eq!((l0[1].a, l0[1].b), (12, 3));

    assert_eq!(r.coactivation(1).len(), 1);
    assert!(
        r.coactivation(99).is_empty(),
        "a layer with no edges is empty, not an error"
    );
}

#[test]
fn duplicate_expert_is_rejected_at_write_time() {
    let f = TempFile::new("dup");
    let mut w = LayoutWriter::create(f.path(), "m").unwrap();
    let k = ExpertKey::new(4, 4);
    w.push_expert(k, Precision::Q4, b"a").unwrap();
    let err = w.push_expert(k, Precision::Q4, b"b").unwrap_err();
    assert!(
        matches!(err, Error::DuplicateExpert(d) if d == k),
        "got {err:?}"
    );
}

#[test]
fn same_expert_index_in_different_layers_is_not_a_duplicate() {
    let f = TempFile::new("layers");
    let mut w = LayoutWriter::create(f.path(), "m").unwrap();
    w.push_expert(ExpertKey::new(3, 5), Precision::Q4, b"early")
        .unwrap();
    w.push_expert(ExpertKey::new(30, 5), Precision::Q4, b"late")
        .unwrap();
    w.finish().unwrap();

    let r = LayoutReader::open(f.path()).unwrap();
    assert_eq!(r.read_expert(ExpertKey::new(3, 5)).unwrap(), b"early");
    assert_eq!(r.read_expert(ExpertKey::new(30, 5)).unwrap(), b"late");
}

#[test]
fn missing_expert_reports_which_one() {
    let f = TempFile::new("missing");
    write_sample(f.path());
    let r = LayoutReader::open(f.path()).unwrap();
    let ghost = ExpertKey::new(7, 7);
    assert!(matches!(r.read_expert(ghost).unwrap_err(), Error::MissingExpert(k) if k == ghost));
}

#[test]
fn model_id_longer_than_the_header_block_is_refused() {
    let f = TempFile::new("longid");
    let too_long = "x".repeat(MODEL_ID_CAPACITY + 1);
    let err = LayoutWriter::create(f.path(), too_long).unwrap_err();
    assert!(matches!(err, Error::ModelIdTooLong(n) if n == MODEL_ID_CAPACITY + 1));

    let f2 = TempFile::new("maxid");
    let exact = "y".repeat(MODEL_ID_CAPACITY);
    let mut w = LayoutWriter::create(f2.path(), exact.clone()).unwrap();
    w.push_expert(ExpertKey::new(0, 0), Precision::Q4, b"z")
        .unwrap();
    w.finish().unwrap();
    assert_eq!(LayoutReader::open(f2.path()).unwrap().model_id(), exact);
}

fn flip_byte(path: &std::path::Path, offset: u64) {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut b = [0u8; 1];
    f.read_exact(&mut b).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(&[b[0] ^ 0xFF]).unwrap();
}

#[test]
fn a_flipped_bit_in_a_payload_is_caught_on_read() {
    let f = TempFile::new("badpayload");
    write_sample(f.path());
    let key = ExpertKey::new(1, 3);
    let at = {
        let r = LayoutReader::open(f.path()).unwrap();
        r.entry(key).unwrap().file_offset(r.header().data_off) + 64
    };
    flip_byte(f.path(), at);

    let r = LayoutReader::open(f.path()).unwrap();
    // opening is still fine, because payloads are not checksummed until read
    let err = r.read_expert(key).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Corrupt {
                region: "expert",
                ..
            }
        ),
        "got {err:?}"
    );
    // every other expert is unaffected, so one bad block does not sink the file
    assert!(r.read_expert(ExpertKey::new(0, 47)).is_ok());
}

#[test]
fn a_flipped_bit_in_the_index_is_caught_at_open() {
    let f = TempFile::new("badindex");
    write_sample(f.path());
    let at = LayoutReader::open(f.path()).unwrap().header().index_off + 8;
    flip_byte(f.path(), at);
    let err = LayoutReader::open(f.path()).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Corrupt {
                region: "index",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn a_flipped_bit_in_the_header_is_caught_before_any_offset_is_used() {
    let f = TempFile::new("badheader");
    write_sample(f.path());
    flip_byte(f.path(), 20);
    let err = LayoutReader::open(f.path()).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Corrupt {
                region: "header",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn a_foreign_file_is_rejected_by_magic_not_by_a_panic() {
    let f = TempFile::new("foreign");
    std::fs::write(f.path(), vec![0xABu8; 8192]).unwrap();
    assert!(matches!(
        LayoutReader::open(f.path()).unwrap_err(),
        Error::BadMagic(_)
    ));
}

#[test]
fn a_truncated_file_reports_where_it_ran_out() {
    let f = TempFile::new("trunc");
    write_sample(f.path());
    let len = std::fs::metadata(f.path()).unwrap().len();
    OpenOptions::new()
        .write(true)
        .open(f.path())
        .unwrap()
        .set_len(len - 16)
        .unwrap();
    let err = LayoutReader::open(f.path()).unwrap_err();
    assert!(matches!(err, Error::Truncated { .. }), "got {err:?}");
}

#[test]
fn header_block_is_exactly_one_alignment_unit() {
    let f = TempFile::new("headerblock");
    write_sample(f.path());
    let r = LayoutReader::open(f.path()).unwrap();
    assert_eq!(r.header().data_off, u64::from(ALIGNMENT));
    assert!(u64::from(r.header().model_id_len) + u64::from(HEADER_LEN) <= u64::from(ALIGNMENT));
}

#[test]
fn an_empty_file_is_valid() {
    let f = TempFile::new("empty");
    let w = LayoutWriter::create(f.path(), "nothing").unwrap();
    assert!(w.is_empty());
    w.finish().unwrap();

    let r = LayoutReader::open(f.path()).unwrap();
    assert_eq!(r.entries().len(), 0);
    assert_eq!(r.payload_bytes(), 0);
}
