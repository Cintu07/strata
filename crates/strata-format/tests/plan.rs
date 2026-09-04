//! read planning: many small wants become few large sequential transfers.
//!
//! these tests are the guard on the claim that makes the storage design work.
//! if coalescing silently stopped happening, correctness would be untouched and
//! throughput would quietly collapse, which is exactly the kind of regression
//! that survives a test suite that only checks bytes.

mod common;

use common::{TempFile, payload};
use strata_format::{ALIGNMENT, ExpertKey, LayoutReader, LayoutWriter, PlanOptions, Precision};

const EXPERT_BYTES: usize = 64 * 1024;

/// eight experts of 64kb each, contiguous on disk in index order.
fn write_run(path: &std::path::Path, n: u32) {
    let mut w = LayoutWriter::create(path, "plan-test").unwrap();
    for e in 0..n {
        w.push_expert(
            ExpertKey::new(0, e),
            Precision::Q4,
            &payload(u64::from(e), EXPERT_BYTES),
        )
        .unwrap();
    }
    w.finish().unwrap();
}

#[test]
fn adjacent_wants_become_one_transfer() {
    let f = TempFile::new("plan-adjacent");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    let keys: Vec<_> = (0..4).map(|e| ExpertKey::new(0, e)).collect();
    let plan = r.plan_reads(&keys, PlanOptions::default()).unwrap();

    assert_eq!(plan.requests.len(), 1, "four adjacent experts are one read");
    assert_eq!(plan.requests[0].wanted.len(), 4);
    assert!(
        plan.requests[0].incidental.is_empty(),
        "nothing was bridged over"
    );
    assert_eq!(plan.transferred_bytes, plan.wanted_bytes);
    assert!((plan.overfetch_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn a_bridged_gap_delivers_the_skipped_experts_for_free() {
    let f = TempFile::new("plan-bridge");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    // want 0 and 3, so 1 and 2 sit in the gap. at 64kb each the gap is 128kb,
    // well inside the default one mebibyte budget.
    let keys = [ExpertKey::new(0, 0), ExpertKey::new(0, 3)];
    let plan = r.plan_reads(&keys, PlanOptions::default()).unwrap();

    assert_eq!(plan.requests.len(), 1);
    let req = &plan.requests[0];
    assert_eq!(req.wanted, vec![ExpertKey::new(0, 0), ExpertKey::new(0, 3)]);
    assert_eq!(
        req.incidental,
        vec![ExpertKey::new(0, 1), ExpertKey::new(0, 2)]
    );

    // the plan is honest about paying for it
    assert_eq!(plan.wanted_bytes, 2 * EXPERT_BYTES as u64);
    assert_eq!(plan.transferred_bytes, 4 * EXPERT_BYTES as u64);
    assert!((plan.overfetch_ratio() - 2.0).abs() < 1e-9);
    assert_eq!(plan.incidental().count(), 2);
}

#[test]
fn a_gap_over_budget_splits_into_two_transfers() {
    let f = TempFile::new("plan-split");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    let keys = [ExpertKey::new(0, 0), ExpertKey::new(0, 3)];
    let tight = PlanOptions {
        max_gap_bytes: 64 * 1024,
        ..PlanOptions::default()
    };
    let plan = r.plan_reads(&keys, tight).unwrap();

    assert_eq!(plan.requests.len(), 2, "a 128kb gap exceeds a 64kb budget");
    assert_eq!(
        plan.transferred_bytes, plan.wanted_bytes,
        "nothing extra was read"
    );
    assert_eq!(plan.incidental().count(), 0);
}

#[test]
fn per_expert_mode_is_the_baseline_the_others_are_measured_against() {
    let f = TempFile::new("plan-baseline");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    let keys: Vec<_> = (0..8).map(|e| ExpertKey::new(0, e)).collect();
    let baseline = r.plan_reads(&keys, PlanOptions::per_expert()).unwrap();
    let coalesced = r.plan_reads(&keys, PlanOptions::default()).unwrap();

    assert_eq!(
        baseline.requests.len(),
        8,
        "one read per expert, the naive behaviour"
    );
    assert_eq!(coalesced.requests.len(), 1);
    // the same bytes either way, in eight requests or in one. the difference is
    // eight queue round trips, which is the whole argument for the layout.
    assert_eq!(baseline.transferred_bytes, coalesced.transferred_bytes);
}

#[test]
fn no_overfetch_still_joins_touching_reads_because_that_is_free() {
    let f = TempFile::new("plan-touching");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    let contiguous: Vec<_> = (0..4).map(|e| ExpertKey::new(0, e)).collect();
    let plan = r
        .plan_reads(&contiguous, PlanOptions::no_overfetch())
        .unwrap();
    assert_eq!(plan.requests.len(), 1);
    assert_eq!(plan.transferred_bytes, plan.wanted_bytes);

    // but it will not bridge a gap, however small
    let split = [ExpertKey::new(0, 0), ExpertKey::new(0, 2)];
    let plan = r.plan_reads(&split, PlanOptions::no_overfetch()).unwrap();
    assert_eq!(plan.requests.len(), 2);
    assert_eq!(plan.transferred_bytes, plan.wanted_bytes);
}

#[test]
fn the_request_cap_bounds_every_transfer() {
    let f = TempFile::new("plan-cap");
    write_run(f.path(), 16);
    let r = LayoutReader::open(f.path()).unwrap();

    let keys: Vec<_> = (0..16).map(|e| ExpertKey::new(0, e)).collect();
    let capped = PlanOptions {
        max_request_bytes: 256 * 1024,
        ..PlanOptions::default()
    };
    let plan = r.plan_reads(&keys, capped).unwrap();

    assert_eq!(
        plan.requests.len(),
        4,
        "16 x 64kb under a 256kb cap is four reads"
    );
    for req in &plan.requests {
        assert!(
            req.len <= capped.max_request_bytes,
            "transfer of {} exceeds the cap",
            req.len
        );
    }
    assert_eq!(plan.transferred_bytes, plan.wanted_bytes);
}

#[test]
fn requests_are_issued_in_disk_order_whatever_order_they_were_asked_in() {
    let f = TempFile::new("plan-order");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    let scrambled = [
        ExpertKey::new(0, 7),
        ExpertKey::new(0, 1),
        ExpertKey::new(0, 4),
        ExpertKey::new(0, 1),
    ];
    let plan = r.plan_reads(&scrambled, PlanOptions::per_expert()).unwrap();

    assert_eq!(plan.requests.len(), 3, "the repeated key is asked for once");
    for pair in plan.requests.windows(2) {
        assert!(
            pair[0].file_offset < pair[1].file_offset,
            "reads must sweep forward"
        );
    }
    assert_eq!(plan.wanted_bytes, 3 * EXPERT_BYTES as u64);
}

#[test]
fn every_transfer_is_alignment_friendly() {
    let f = TempFile::new("plan-align");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    let keys: Vec<_> = (0..8).step_by(3).map(|e| ExpertKey::new(0, e)).collect();
    let plan = r.plan_reads(&keys, PlanOptions::default()).unwrap();
    for req in &plan.requests {
        assert_eq!(
            req.file_offset % u64::from(ALIGNMENT),
            0,
            "direct io rejects unaligned offsets"
        );
        assert_eq!(
            req.len % u64::from(ALIGNMENT),
            0,
            "direct io rejects unaligned lengths"
        );
    }
}

#[test]
fn slicing_a_transfer_reproduces_the_individual_payloads() {
    let f = TempFile::new("plan-slice");
    write_run(f.path(), 8);
    let r = LayoutReader::open(f.path()).unwrap();

    let keys = [ExpertKey::new(0, 1), ExpertKey::new(0, 4)];
    let plan = r.plan_reads(&keys, PlanOptions::default()).unwrap();
    assert_eq!(plan.requests.len(), 1);

    let req = &plan.requests[0];
    let buf = r.execute(req).unwrap();
    assert_eq!(buf.len() as u64, req.len);

    // the wanted experts slice out byte for byte
    for k in req.wanted.iter().chain(req.incidental.iter()) {
        let entry = r.entry(*k).unwrap();
        let range = req
            .slice_of(entry)
            .expect("entry lies inside its own transfer");
        assert_eq!(
            &buf[range],
            &r.read_expert(*k).unwrap()[..],
            "slice mismatch for {k}"
        );
    }
}

#[test]
fn planning_an_absent_expert_fails_rather_than_reading_the_wrong_bytes() {
    let f = TempFile::new("plan-absent");
    write_run(f.path(), 4);
    let r = LayoutReader::open(f.path()).unwrap();
    let err = r.plan_reads(
        &[ExpertKey::new(0, 0), ExpertKey::new(9, 9)],
        PlanOptions::default(),
    );
    assert!(matches!(err, Err(strata_format::Error::MissingExpert(_))));
}

#[test]
fn an_empty_want_list_plans_nothing() {
    let f = TempFile::new("plan-none");
    write_run(f.path(), 4);
    let r = LayoutReader::open(f.path()).unwrap();
    let plan = r.plan_reads(&[], PlanOptions::default()).unwrap();
    assert!(plan.requests.is_empty());
    assert_eq!(plan.transferred_bytes, 0);
    assert!(
        (plan.overfetch_ratio() - 1.0).abs() < f64::EPSILON,
        "no wants is not infinite overfetch"
    );
}

/// the headline claim of the layout, stated as a test: with experts placed in
/// co-activation order, a realistically sparse want set still sweeps the file
/// in a handful of reads rather than one read per expert.
#[test]
fn a_sparse_want_set_over_a_co_activation_ordered_file_stays_sequential() {
    let f = TempFile::new("plan-sweep");
    write_run(f.path(), 128);
    let r = LayoutReader::open(f.path()).unwrap();

    // top 8 of 128 routed, but clustered, which is what the ordering pass is for
    let keys: Vec<_> = [2u32, 3, 5, 40, 41, 43, 90, 91]
        .iter()
        .map(|&e| ExpertKey::new(0, e))
        .collect();

    let baseline = r.plan_reads(&keys, PlanOptions::per_expert()).unwrap();
    let plan = r.plan_reads(&keys, PlanOptions::default()).unwrap();

    assert_eq!(baseline.requests.len(), 8);
    assert_eq!(plan.requests.len(), 3, "three clusters become three reads");
    assert!(
        plan.overfetch_ratio() < 1.5,
        "ratio was {}",
        plan.overfetch_ratio()
    );
}
