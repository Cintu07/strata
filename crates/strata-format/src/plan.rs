//! turning a set of wanted experts into a small number of large reads.
//!
//! a consumer gen4 nvme does five to seven gigabytes a second sequentially and
//! under a tenth of a gigabyte a second on random 4k at queue depth one. the
//! difference is not a tuning constant, it is two orders of magnitude, and it
//! is the reason this module exists.
//!
//! the trick is that a read which has already started is cheap to make longer.
//! if two wanted experts are near each other on disk, bridging the gap between
//! them transfers bytes nobody asked for but saves the fixed cost of a second
//! request. at roughly 100 microseconds of request latency and 6 GB/s of
//! streaming bandwidth, that fixed cost is worth about 600kb of bytes, so the
//! default gap budget of one mebibyte is the right order of magnitude and is
//! deliberately a little generous: the bridged bytes are not waste, they are
//! co-activated experts the layout pass put there on purpose, and they land in
//! the cache for free.
//!
//! measure [`ReadPlan::overfetch_ratio`] against the achieved bandwidth on the
//! target device before changing the defaults.

use crate::types::{ExpertEntry, ExpertKey};

/// knobs for [`crate::LayoutReader::plan_reads`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOptions {
    /// largest run of unwanted bytes worth bridging rather than paying for a
    /// second request.
    pub max_gap_bytes: u64,
    /// cap on a single transfer, which bounds the staging buffer and keeps one
    /// enormous read from monopolising the queue.
    pub max_request_bytes: u64,
    /// when false, every expert gets its own transfer no matter how they sit on
    /// disk. only useful as a measurement baseline.
    pub coalesce: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            max_gap_bytes: 1 << 20,
            max_request_bytes: 32 << 20,
            coalesce: true,
        }
    }
}

impl PlanOptions {
    /// merge only experts that are already physically adjacent, so the transfer
    /// is strictly the bytes that were asked for.
    ///
    /// this is not the same as refusing to coalesce. joining two touching reads
    /// costs nothing and saves a request, so it is always worth doing. use this
    /// when the ram budget cannot absorb any overfetch at all.
    #[must_use]
    pub const fn no_overfetch() -> Self {
        Self {
            max_gap_bytes: 0,
            max_request_bytes: u64::MAX,
            coalesce: true,
        }
    }

    /// one transfer per expert. the baseline the other two modes are measured
    /// against, and the thing a naive implementation does by accident.
    #[must_use]
    pub const fn per_expert() -> Self {
        Self {
            max_gap_bytes: 0,
            max_request_bytes: u64::MAX,
            coalesce: false,
        }
    }
}

/// one contiguous transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadRequest {
    /// absolute offset in the file, aligned.
    pub file_offset: u64,
    /// bytes to transfer, aligned.
    pub len: u64,
    /// offset of `file_offset` within the data region, used to slice results.
    pub data_offset: u64,
    /// experts the caller asked for that this transfer satisfies.
    pub wanted: Vec<ExpertKey>,
    /// experts that fell inside a bridged gap and came along at no extra cost.
    pub incidental: Vec<ExpertKey>,
}

impl ReadRequest {
    /// byte range of one expert's payload within the buffer this request fills.
    ///
    /// returns `None` if the entry is not inside this transfer.
    #[must_use]
    pub fn slice_of(&self, entry: &ExpertEntry) -> Option<std::ops::Range<usize>> {
        let start = entry.offset.checked_sub(self.data_offset)?;
        let end = start + entry.len;
        (end <= self.len).then_some(start as usize..end as usize)
    }
}

/// the planned transfers plus what the plan cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadPlan {
    /// transfers in ascending disk order, which is the order to issue them in.
    pub requests: Vec<ReadRequest>,
    /// aligned bytes belonging to experts the caller actually asked for.
    pub wanted_bytes: u64,
    /// bytes that will cross the pcie bus and the nvme link.
    pub transferred_bytes: u64,
}

impl ReadPlan {
    /// bytes transferred per byte wanted. 1.0 is a perfect plan, and anything
    /// under about 1.3 is usually a better trade than the requests it saved.
    #[must_use]
    pub fn overfetch_ratio(&self) -> f64 {
        if self.wanted_bytes == 0 {
            return 1.0;
        }
        self.transferred_bytes as f64 / self.wanted_bytes as f64
    }

    /// experts that arrive for free because they sat inside a bridged gap.
    /// these are the co-activation layout pass paying for itself.
    pub fn incidental(&self) -> impl Iterator<Item = ExpertKey> + '_ {
        self.requests
            .iter()
            .flat_map(|r| r.incidental.iter().copied())
    }

    pub(crate) fn build(
        entries: &[ExpertEntry],
        idx: &mut Vec<usize>,
        data_off: u64,
        opts: PlanOptions,
    ) -> Self {
        idx.sort_unstable();
        idx.dedup();

        let mut requests: Vec<ReadRequest> = Vec::new();
        let mut wanted_bytes = 0u64;
        let mut transferred_bytes = 0u64;
        let mut open: Option<(usize, usize)> = None; // (first entry index, last entry index)

        for &i in idx.iter() {
            let e = &entries[i];
            wanted_bytes += e.padded_len();

            match open {
                Some((first, last)) => {
                    let start = entries[first].offset;
                    let cur_end = entries[last].offset + entries[last].padded_len();
                    let new_end = e.offset + e.padded_len();
                    let gap = e.offset.saturating_sub(cur_end);
                    if opts.coalesce
                        && gap <= opts.max_gap_bytes
                        && new_end - start <= opts.max_request_bytes
                    {
                        open = Some((first, i));
                        continue;
                    }
                    let req = finish_request(entries, first, last, idx, data_off);
                    transferred_bytes += req.len;
                    requests.push(req);
                    open = Some((i, i));
                }
                None => open = Some((i, i)),
            }
        }

        if let Some((first, last)) = open {
            let req = finish_request(entries, first, last, idx, data_off);
            transferred_bytes += req.len;
            requests.push(req);
        }

        Self {
            requests,
            wanted_bytes,
            transferred_bytes,
        }
    }
}

/// close out one transfer spanning entry indices `first..=last`.
///
/// because `entries` is sorted by offset, every entry index in that range lies
/// physically inside the transfer, so the ones that were not requested are
/// exactly the incidental set. no offset arithmetic is needed to find them.
fn finish_request(
    entries: &[ExpertEntry],
    first: usize,
    last: usize,
    wanted_idx: &[usize],
    data_off: u64,
) -> ReadRequest {
    let start = entries[first].offset;
    let end = entries[last].offset + entries[last].padded_len();
    let mut wanted = Vec::new();
    let mut incidental = Vec::new();
    for (i, e) in entries.iter().enumerate().take(last + 1).skip(first) {
        if wanted_idx.binary_search(&i).is_ok() {
            wanted.push(e.key);
        } else {
            incidental.push(e.key);
        }
    }
    ReadRequest {
        file_offset: data_off + start,
        len: end - start,
        data_offset: start,
        wanted,
        incidental,
    }
}
