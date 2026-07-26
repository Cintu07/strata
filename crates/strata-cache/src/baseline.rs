//! the policies strata has to beat, implemented here so that beating them is a
//! test rather than a sentence in a readme.
//!
//! every offloading system in this space reports against lru, and most report
//! against nothing else. having lru and an offline optimum in the same crate,
//! driven by the same trace through the same interface, is what makes a hit
//! rate number mean something: it puts a floor and a ceiling around it.

use crate::cache::{ExpertCache, ExpertDesc};
use crate::config::CacheConfig;
use crate::stats::CacheStats;
use std::collections::{HashMap, HashSet, VecDeque};
use strata_format::ExpertKey;

/// an online cache policy driven one access at a time.
pub trait Policy {
    /// name for reports.
    fn name(&self) -> &'static str;

    /// look up an expert and, on a miss, fetch it and offer it to the policy.
    /// returns whether it was a hit.
    fn access(&mut self, key: ExpertKey, desc: ExpertDesc) -> bool;

    /// bytes currently held.
    fn resident_bytes(&self) -> u64;

    /// counters.
    fn stats(&self) -> &CacheStats;
}

impl Policy for ExpertCache<()> {
    fn name(&self) -> &'static str {
        "strata"
    }

    fn access(&mut self, key: ExpertKey, desc: ExpertDesc) -> bool {
        if self.get(key).is_some() {
            return true;
        }
        self.admit(key, desc, ());
        false
    }

    fn resident_bytes(&self) -> u64 {
        Self::resident_bytes(self)
    }

    fn stats(&self) -> &CacheStats {
        Self::stats(self)
    }
}

/// least recently used, admitting on first touch.
///
/// the baseline every expert offloading system reports against. it is not a
/// straw man: recency is a genuinely strong signal inside a single paragraph of
/// generation. what it cannot do is survive a topic switch, because it has no
/// memory of which experts a domain used before the digression.
#[derive(Debug)]
pub struct LruCache {
    capacity_bytes: u64,
    resident_bytes: u64,
    sizes: HashMap<ExpertKey, u64>,
    order: VecDeque<ExpertKey>,
    stats: CacheStats,
}

impl LruCache {
    /// an lru cache of the given size.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            sizes: HashMap::new(),
            order: VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    fn touch(&mut self, key: ExpertKey) {
        if let Some(pos) = self.order.iter().position(|&k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
    }
}

impl Policy for LruCache {
    fn name(&self) -> &'static str {
        "lru"
    }

    fn access(&mut self, key: ExpertKey, desc: ExpertDesc) -> bool {
        if self.sizes.contains_key(&key) {
            self.stats.hits += 1;
            self.touch(key);
            return true;
        }
        self.stats.misses += 1;
        self.stats.bytes_missed += desc.stored_bytes;

        if desc.stored_bytes > self.capacity_bytes {
            self.stats.rejected_oversized += 1;
            return false;
        }
        while self.resident_bytes + desc.stored_bytes > self.capacity_bytes {
            let Some(victim) = self.order.pop_front() else {
                break;
            };
            if let Some(bytes) = self.sizes.remove(&victim) {
                self.resident_bytes -= bytes;
                self.stats.evictions += 1;
                self.stats.bytes_evicted += bytes;
            }
        }
        self.sizes.insert(key, desc.stored_bytes);
        self.order.push_back(key);
        self.resident_bytes += desc.stored_bytes;
        self.stats.admissions += 1;
        self.stats.bytes_admitted += desc.stored_bytes;
        false
    }

    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// least frequently used over the whole run, with no decay.
///
/// this is the baseline that actually matters on a real decoder trace, and it
/// was missing from this crate for a while, which flattered strata.
///
/// lru is the policy every paper in this space reports against, but on a real
/// routing trace lru is not a competitor, it is a pathology: a token touches
/// `n_layers * top_k` distinct pairs, so any cache smaller than that sees a
/// cyclic scan and lru returns exactly zero. beating a policy that scores zero
/// proves nothing about the eviction score. frequency is what survives the scan
/// and therefore what strata has to actually beat.
///
/// no decay at all, so it ossifies on a topic switch. that is a real failure
/// mode and it is left in rather than tuned away, because a baseline that has
/// been tuned is not a baseline.
#[derive(Debug)]
pub struct LfuCache {
    capacity_bytes: u64,
    resident_bytes: u64,
    freq: HashMap<ExpertKey, u64>,
    sizes: HashMap<ExpertKey, u64>,
    stats: CacheStats,
}

impl LfuCache {
    /// an lfu cache of the given size.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            freq: HashMap::new(),
            sizes: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// least frequent resident key, ties broken by packed key so the result
    /// does not depend on hash iteration order.
    fn victim(&self) -> Option<ExpertKey> {
        self.sizes
            .keys()
            .min_by_key(|k| (self.freq.get(k).copied().unwrap_or(0), k.packed()))
            .copied()
    }
}

impl Policy for LfuCache {
    fn name(&self) -> &'static str {
        "lfu"
    }

    fn access(&mut self, key: ExpertKey, desc: ExpertDesc) -> bool {
        *self.freq.entry(key).or_insert(0) += 1;

        if self.sizes.contains_key(&key) {
            self.stats.hits += 1;
            return true;
        }
        self.stats.misses += 1;
        self.stats.bytes_missed += desc.stored_bytes;

        if desc.stored_bytes > self.capacity_bytes {
            self.stats.rejected_oversized += 1;
            return false;
        }

        let incoming = self.freq.get(&key).copied().unwrap_or(0);
        while self.resident_bytes + desc.stored_bytes > self.capacity_bytes {
            let Some(victim) = self.victim() else {
                break;
            };
            // an incoming key that is rarer than the rarest resident does not
            // earn the slot. without this, lfu degenerates to a scan-driven
            // cache on exactly the traces frequency is supposed to survive.
            if self.freq.get(&victim).copied().unwrap_or(0) > incoming {
                return false;
            }
            if let Some(bytes) = self.sizes.remove(&victim) {
                self.resident_bytes -= bytes;
                self.stats.evictions += 1;
                self.stats.bytes_evicted += bytes;
            }
        }

        self.sizes.insert(key, desc.stored_bytes);
        self.resident_bytes += desc.stored_bytes;
        self.stats.admissions += 1;
        self.stats.bytes_admitted += desc.stored_bytes;
        false
    }

    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// strata's policy with admission control switched off, so that the effect of
/// admission can be separated from the effect of the eviction score.
///
/// a comparison that changes two things at once explains nothing.
#[must_use]
pub fn without_admission(capacity_bytes: u64) -> ExpertCache<()> {
    ExpertCache::new(CacheConfig {
        tinylfu_admission: false,
        ..CacheConfig::with_capacity(capacity_bytes)
    })
}

/// offline optimum: evict whatever is used furthest in the future.
///
/// this is belady's min. it is not implementable online, and that is the point:
/// it is the ceiling that says how much of a miss rate is the policy's fault
/// and how much is simply the cache being too small for the workload.
///
/// # the caveat that matters
///
/// min is exactly optimal only when every object is the same size. with mixed
/// sizes, optimal caching is a knapsack problem and this becomes an
/// approximation rather than a true bound. within one layer experts are
/// uniform, so on a single layer trace the number is exact; across layers of
/// differing width, read it as a close estimate of the ceiling and not as a
/// proof.
#[derive(Debug)]
pub struct BeladyOracle {
    capacity_bytes: u64,
}

/// what an oracle run measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OracleResult {
    /// lookups served from ram.
    pub hits: u64,
    /// lookups that went to disk.
    pub misses: u64,
    /// bytes read from disk.
    pub bytes_missed: u64,
}

impl OracleResult {
    /// fraction of lookups served from ram.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

impl BeladyOracle {
    /// an oracle for a cache of the given size.
    #[must_use]
    pub const fn new(capacity_bytes: u64) -> Self {
        Self { capacity_bytes }
    }

    /// run the whole trace and report the best any cache of this size could
    /// have done.
    ///
    /// `sizes` gives the stored size of each expert. an expert absent from the
    /// map is treated as absent from the model and skipped.
    #[must_use]
    pub fn run(&self, trace: &[ExpertKey], sizes: &HashMap<ExpertKey, u64>) -> OracleResult {
        // every position each expert appears at, so the next use of a resident
        // expert is a lookup rather than a scan
        let mut positions: HashMap<ExpertKey, Vec<usize>> = HashMap::new();
        for (i, &key) in trace.iter().enumerate() {
            positions.entry(key).or_default().push(i);
        }
        // cursor[k] indexes the first position of k not yet consumed. it is
        // advanced when k is accessed, so for any resident expert it already
        // points strictly past the current step.
        let mut cursor: HashMap<ExpertKey, usize> = HashMap::new();

        let mut resident: HashSet<ExpertKey> = HashSet::new();
        let mut resident_bytes = 0u64;
        let mut out = OracleResult {
            hits: 0,
            misses: 0,
            bytes_missed: 0,
        };

        let next_use = |k: ExpertKey, cursor: &HashMap<ExpertKey, usize>| -> usize {
            let c = cursor.get(&k).copied().unwrap_or(0);
            positions
                .get(&k)
                .and_then(|p| p.get(c))
                .copied()
                .unwrap_or(usize::MAX)
        };

        for (i, &key) in trace.iter().enumerate() {
            let Some(&bytes) = sizes.get(&key) else {
                continue;
            };
            debug_assert_eq!(positions[&key][cursor.get(&key).copied().unwrap_or(0)], i);
            *cursor.entry(key).or_insert(0) += 1;

            if resident.contains(&key) {
                out.hits += 1;
                continue;
            }
            out.misses += 1;
            out.bytes_missed += bytes;
            if bytes > self.capacity_bytes {
                continue;
            }

            while resident_bytes + bytes > self.capacity_bytes {
                // evict whichever resident expert is needed furthest away,
                // breaking ties on the key so a replay is reproducible
                let victim = resident
                    .iter()
                    .copied()
                    .max_by_key(|&k| (next_use(k, &cursor), k.packed()));
                let Some(victim) = victim else { break };
                resident.remove(&victim);
                resident_bytes -= sizes.get(&victim).copied().unwrap_or(0);
            }
            resident.insert(key);
            resident_bytes += bytes;
        }
        out
    }
}
