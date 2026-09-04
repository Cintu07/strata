//! a decaying frequency estimator.
//!
//! # why an estimator and not a counter
//!
//! the first version of this cache kept a lifetime access count on each
//! resident entry and admitted a new expert only if it outscored the entry it
//! would evict. that deadlocks, and the measurement in `tests/measure.rs`
//! showed it plainly: after one topic has run for a while its experts carry
//! counts in the tens, every newcomer arrives with a count of one, so nothing
//! is ever admitted, so nothing is ever evicted, so the aging clock never
//! advances and the cache freezes on whatever it happened to see first. on a
//! four domain workload that scored 0.25 against lru's 0.96.
//!
//! the flaw is comparing a lifetime count against a first impression. both
//! sides have to be measured over the same recent window, and the window has to
//! include accesses that missed, because an expert coming back after a
//! digression has plenty of recent history and none of it is resident.
//!
//! so: a count-min sketch over every access, hit or miss, whose counters are
//! halved whenever the sample fills. that is tinylfu, and the halving is the
//! recency decay the prd asks for. it costs eight kilobytes for four thousand
//! experts, which is nothing against the ram this system is fighting for.
//!
//! # accuracy
//!
//! four bit counters saturate at fifteen. that is deliberate: the question
//! being asked is only ever "is this one used more than that one", and past
//! fifteen in a window the answer does not change. counters are shared across
//! keys so estimates are biased upward, never downward, and taking the minimum
//! across four independent rows makes a large overestimate unlikely.

/// four bit count-min sketch with periodic halving.
#[derive(Debug)]
pub(crate) struct FrequencySketch {
    /// sixteen four bit counters per word.
    table: Vec<u64>,
    /// counters per row, a power of two so indexing is a mask.
    counters: usize,
    /// increments since the last halving.
    size: u64,
    /// increments that fill the window.
    sample_size: u64,
}

const ROWS: usize = 4;
const MAX_COUNT: u64 = 15;

/// one odd seed per row. any distinct odd constants work; these are the
/// fractional bits of common irrationals, which is the usual way to pick
/// constants nobody chose for a reason.
const SEEDS: [u64; ROWS] = [
    0x9E37_79B9_7F4A_7C15,
    0xBF58_476D_1CE4_E5B9,
    0x94D0_49BB_1331_11EB,
    0xC2B2_AE3D_27D4_EB4F,
];

impl FrequencySketch {
    /// a sketch whose counter table is sized for roughly `universe` distinct
    /// keys, and whose decay window is sized for a cache holding
    /// `cache_items` of them.
    ///
    /// those two numbers are different and confusing them is a real bug rather
    /// than a tuning miss. the table has to be large enough that unrelated
    /// experts across the whole model rarely share a counter, so it is sized by
    /// the universe. the decay window has to be short enough that a topic which
    /// stopped being used fades before the cache has turned over, so it is
    /// sized by the cache. sizing the window by the universe was the second
    /// version of the deadlock described at the top of this file: on a model
    /// with four thousand experts and a cache holding twelve, the counters
    /// never halved inside a whole run and the first topic seen kept the cache
    /// forever.
    pub(crate) fn new(universe: usize, cache_items: usize) -> Self {
        let counters = universe.max(64).next_power_of_two();
        let words = (counters * ROWS).div_ceil(16);
        let mut s = Self {
            table: vec![0; words],
            counters,
            size: 0,
            sample_size: 0,
        };
        s.set_cache_items(cache_items);
        s
    }

    /// retune the decay window as the cache learns how big experts actually are.
    ///
    /// ten passes over a full cache before the counters halve. long enough that
    /// a stable working set builds a clear lead over one shot traffic, short
    /// enough that a domain which fell out of use loses its claim within a few
    /// hundred accesses rather than never.
    pub(crate) fn set_cache_items(&mut self, cache_items: usize) {
        self.sample_size = 10 * cache_items.max(1) as u64;
    }

    /// record one access. call this on every access, including misses.
    pub(crate) fn increment(&mut self, key: u64) {
        let mut changed = false;
        for row in 0..ROWS {
            let idx = self.index(key, row);
            if self.bump(idx) {
                changed = true;
            }
        }
        if changed {
            self.size += 1;
            if self.size >= self.sample_size {
                self.halve();
            }
        }
    }

    /// estimated recent access count, saturating at fifteen.
    pub(crate) fn estimate(&self, key: u64) -> u64 {
        (0..ROWS)
            .map(|row| self.read(self.index(key, row)))
            .min()
            .unwrap_or(0)
    }

    /// forget everything, for a hard context switch.
    pub(crate) fn clear(&mut self) {
        self.table.fill(0);
        self.size = 0;
    }

    fn index(&self, key: u64, row: usize) -> usize {
        let h = splitmix64(key ^ SEEDS[row]);
        row * self.counters + (h as usize & (self.counters - 1))
    }

    fn read(&self, idx: usize) -> u64 {
        let shift = (idx & 15) * 4;
        (self.table[idx >> 4] >> shift) & 0xF
    }

    /// returns whether the counter actually moved, so a saturated counter does
    /// not consume window budget.
    fn bump(&mut self, idx: usize) -> bool {
        let shift = (idx & 15) * 4;
        let word = &mut self.table[idx >> 4];
        if (*word >> shift) & 0xF >= MAX_COUNT {
            return false;
        }
        *word += 1 << shift;
        true
    }

    /// halve every counter at once.
    ///
    /// shifting the whole word right by one bleeds each nibble's low bit into
    /// the top of the nibble below, so the mask clears those carried bits.
    fn halve(&mut self) {
        for w in &mut self.table {
            *w = (*w >> 1) & 0x7777_7777_7777_7777;
        }
        self.size /= 2;
    }
}

const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::{FrequencySketch, MAX_COUNT};

    #[test]
    fn an_unseen_key_estimates_zero() {
        let s = FrequencySketch::new(256, 64);
        assert_eq!(s.estimate(42), 0);
    }

    #[test]
    fn counts_rise_with_use() {
        let mut s = FrequencySketch::new(256, 64);
        for _ in 0..5 {
            s.increment(7);
        }
        assert!(s.estimate(7) >= 5, "count-min never underestimates");
    }

    #[test]
    fn counters_saturate_rather_than_wrap() {
        let mut s = FrequencySketch::new(256, 64);
        for _ in 0..1000 {
            s.increment(7);
        }
        assert_eq!(s.estimate(7), MAX_COUNT);
    }

    #[test]
    fn a_hot_key_outranks_a_cold_one() {
        let mut s = FrequencySketch::new(1024, 256);
        for i in 0..2000u64 {
            s.increment(i);
            if i % 3 == 0 {
                s.increment(1);
            }
        }
        assert!(s.estimate(1) > s.estimate(1999));
    }

    #[test]
    fn halving_lets_a_stale_key_fade_below_a_fresh_one() {
        let mut s = FrequencySketch::new(64, 8);
        for _ in 0..MAX_COUNT {
            s.increment(1);
        }
        assert_eq!(s.estimate(1), MAX_COUNT);

        // key 1 is never touched again while a long run of other traffic goes
        // through, which is what a topic switch looks like
        for i in 0..4000u64 {
            s.increment(100 + i);
        }
        for _ in 0..8 {
            s.increment(2);
        }
        assert!(
            s.estimate(2) > s.estimate(1),
            "stale {} should have decayed below fresh {}",
            s.estimate(1),
            s.estimate(2)
        );
    }

    #[test]
    fn clear_resets_everything() {
        let mut s = FrequencySketch::new(64, 8);
        for _ in 0..10 {
            s.increment(5);
        }
        s.clear();
        assert_eq!(s.estimate(5), 0);
    }
}
