//! counters, because the hit rate is the product.
//!
//! g2 in the prd is a number this struct reports. every field here exists to
//! make a claim in the readme checkable, or to explain a hit rate that came out
//! lower than expected.

/// running totals for one cache instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// lookups satisfied from ram.
    pub hits: u64,
    /// lookups that had to go to nvme.
    pub misses: u64,
    /// experts accepted into the cache.
    pub admissions: u64,
    /// window victims that failed to outrank the main region entry they would
    /// have displaced, and left the cache instead.
    pub contention_losses: u64,
    /// window victims that earned a place in the main region.
    pub window_promotions: u64,
    /// experts refused because they do not fit in the cache at any price.
    pub rejected_oversized: u64,
    /// experts evicted to make room.
    pub evictions: u64,
    /// bytes that entered the cache.
    pub bytes_admitted: u64,
    /// bytes that left it.
    pub bytes_evicted: u64,
    /// bytes that had to be read from nvme, which is the metric that most
    /// directly reflects what the system is doing and the one nobody reports.
    pub bytes_missed: u64,
    /// experts promoted to dequantised residency.
    pub promotions: u64,
}

impl CacheStats {
    /// fraction of lookups served from ram, in `[0, 1]`.
    ///
    /// returns 0 before any lookup, which is the honest answer to a question
    /// that has not been asked yet.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// total lookups.
    #[must_use]
    pub const fn lookups(&self) -> u64 {
        self.hits + self.misses
    }

    /// average bytes read from nvme per lookup.
    #[must_use]
    pub fn bytes_missed_per_lookup(&self) -> f64 {
        let total = self.lookups();
        if total == 0 {
            return 0.0;
        }
        self.bytes_missed as f64 / total as f64
    }

    /// how much of what came in has already gone out again.
    ///
    /// a churn ratio near one means the cache is admitting and evicting the
    /// same working set repeatedly, which is the signature of a capacity that
    /// sits just under the knee rather than a policy that is behaving badly.
    #[must_use]
    pub fn churn_ratio(&self) -> f64 {
        if self.bytes_admitted == 0 {
            return 0.0;
        }
        self.bytes_evicted as f64 / self.bytes_admitted as f64
    }
}
