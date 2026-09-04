//! the expert cache itself.

use crate::config::CacheConfig;
use crate::sketch::FrequencySketch;
use crate::stats::CacheStats;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use strata_format::ExpertKey;

/// how an expert is being held in ram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// stored exactly as it is on disk. cheap in bytes, and every use pays a
    /// dequantisation pass.
    Quantized,
    /// already dequantised. larger, and free at the point of use.
    Dequantized,
}

/// which region of the cache an expert is sitting in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// the probationary window. everything enters here.
    Window,
    /// the protected main region, ordered by eviction score.
    Main,
}

/// what the caller knows about an expert when offering it to the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertDesc {
    /// bytes the expert occupies in its on-disk form, which is what a reload
    /// would have to transfer.
    pub stored_bytes: u64,
    /// bytes it would occupy dequantised. equal to `stored_bytes` for a
    /// precision that needs no dequantisation.
    pub dequantized_bytes: u64,
}

impl ExpertDesc {
    /// an expert that is the same size either way.
    #[must_use]
    pub const fn plain(bytes: u64) -> Self {
        Self {
            stored_bytes: bytes,
            dequantized_bytes: bytes,
        }
    }

    const fn bytes_for(&self, r: Residency) -> u64 {
        match r {
            Residency::Quantized => self.stored_bytes,
            Residency::Dequantized => self.dequantized_bytes,
        }
    }
}

/// the outcome of offering an expert to the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// resident now, in the probationary window. whether it survives depends on
    /// what it does next, which is the point of the window.
    Admitted,
    /// larger than the whole cache, so it cannot be held at any price.
    RejectedOversized,
}

impl Admission {
    /// whether the expert is resident afterwards.
    #[must_use]
    pub const fn admitted(self) -> bool {
        matches!(self, Self::Admitted)
    }
}

#[derive(Debug)]
struct Entry<T> {
    value: T,
    desc: ExpertDesc,
    residency: Residency,
    region: Region,
    /// current eviction score, only meaningful in the main region.
    score: f64,
    /// bumped on every score change so stale heap items can be discarded.
    version: u64,
}

impl<T> Entry<T> {
    const fn resident_bytes(&self) -> u64 {
        self.desc.bytes_for(self.residency)
    }
}

/// heap item ordered so that `BinaryHeap` pops the lowest score first.
#[derive(Debug)]
struct Victim {
    score: f64,
    key: ExpertKey,
    version: u64,
}

impl PartialEq for Victim {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Victim {}
impl Ord for Victim {
    fn cmp(&self, other: &Self) -> Ordering {
        // reversed on score so the max-heap yields the minimum, then by key so
        // that ties break deterministically and a replay is reproducible
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.key.cmp(&self.key))
    }
}
impl PartialOrd for Victim {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// a size and cost aware expert cache: a probationary lru window in front of a
/// frequency scored main region, with admission decided by a decaying sketch.
///
/// # the policy, and why it is this one
///
/// three forces have to be balanced, and each one alone fails a workload the
/// others handle.
///
/// **frequency**, because expert access is heavily skewed and domain
/// correlated. a coding conversation hits a stable subset. pure recency throws
/// that structure away at every digression.
///
/// **cost over size**, because experts differ in size and eviction is a
/// knapsack rather than a popularity contest. the main region scores entries by
///
/// ```text
/// score(e) = clock + estimated_freq(e) * reload_seconds(e) / resident_bytes(e)
/// ```
///
/// evicts the lowest, then raises `clock` to whatever it just threw out. that
/// rising floor is greedy dual aging: an expert's advantage erodes on its own,
/// with no sweep over the table. the reload cost includes the fixed request
/// latency, which is what stops the cost term cancelling against size.
///
/// **admission**, because the long tail of a prefill would otherwise evict the
/// working set while contributing nothing. new experts land in a small window
/// and only have to justify themselves when they fall out of it, at which point
/// they are compared against the main region's weakest entry using
/// [`crate::sketch`] estimates on both sides.
///
/// that last detail is the one that matters. an earlier version compared a
/// resident lifetime access count against a newcomer's count of one, which
/// deadlocks: nothing is ever admitted, so nothing is ever evicted, so the
/// clock never advances, and the cache freezes on whatever it saw first. it
/// scored 0.25 against lru's 0.96 on a topic switching workload. both sides
/// have to be measured over the same recent window, and that window has to
/// count misses, because an expert returning after a digression has plenty of
/// recent history and none of it is resident.
///
/// # ownership
///
/// generic over the cached payload so the same code serves both the engine,
/// where `T` is the expert bytes, and trace replay, where `T` is `()` and the
/// cache is a pure simulation of the policy. those are not two implementations
/// that might drift apart, they are one.
///
/// # example
///
/// ```
/// use strata_cache::{CacheConfig, ExpertCache, ExpertDesc};
/// use strata_format::ExpertKey;
///
/// let mut cache: ExpertCache<Vec<u8>> = ExpertCache::new(CacheConfig::with_capacity(1 << 20));
/// let k = ExpertKey::new(3, 17);
/// let desc = ExpertDesc::plain(64 * 1024);
///
/// assert!(cache.get(k).is_none());
/// cache.admit(k, desc, vec![0u8; 64 * 1024]);
/// assert!(cache.get(k).is_some());
/// ```
#[derive(Debug)]
pub struct ExpertCache<T = ()> {
    config: CacheConfig,
    entries: HashMap<ExpertKey, Entry<T>>,
    /// main region eviction order, lowest score first, with lazy invalidation.
    heap: BinaryHeap<Victim>,
    /// window recency order, least recently used at the front.
    window: VecDeque<ExpertKey>,
    sketch: FrequencySketch,
    window_bytes: u64,
    main_bytes: u64,
    clock: f64,
    version: u64,
    /// running mean of admitted expert sizes, used to translate the byte budget
    /// into a number of items so the sketch can size its decay window.
    mean_stored_bytes: f64,
    /// current window budget in bytes. it moves at runtime when the window is
    /// adaptive, which is why this is state and not a config lookup.
    window_budget: u64,
    /// whether the last window adjustment grew the window.
    adapt_grow: bool,
    /// hit rate measured over the previous adaptation interval.
    prev_hit_rate: f64,
    /// hit counter at the last adaptation checkpoint.
    prev_hits: u64,
    /// lookup counter at the last adaptation checkpoint.
    prev_lookups: u64,
    stats: CacheStats,
}

impl<T> ExpertCache<T> {
    /// a cache holding at most `config.capacity_bytes` of experts.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            sketch: FrequencySketch::new(config.expected_experts, 1),
            config,
            entries: HashMap::new(),
            heap: BinaryHeap::new(),
            window: VecDeque::new(),
            window_bytes: 0,
            main_bytes: 0,
            clock: 0.0,
            version: 0,
            mean_stored_bytes: 0.0,
            window_budget: config.window_bytes(),
            adapt_grow: true,
            prev_hit_rate: -1.0,
            prev_hits: 0,
            prev_lookups: 0,
            stats: CacheStats::default(),
        }
    }

    /// look one expert up, counting the hit or miss.
    ///
    /// every lookup feeds the frequency sketch, hit or miss. the misses are the
    /// important half: they are how an expert that is about to come back into
    /// fashion accumulates the evidence it needs to be let in.
    pub fn get(&mut self, key: ExpertKey) -> Option<&T> {
        self.sketch.increment(key.packed());

        let Some(e) = self.entries.get(&key) else {
            self.stats.misses += 1;
            return None;
        };
        self.stats.hits += 1;

        match e.region {
            Region::Window => self.touch_window(key),
            Region::Main => self.rescore(key),
        }
        self.maybe_promote(key);
        self.entries.get(&key).map(|e| &e.value)
    }

    /// whether an expert is resident, without counting a lookup or feeding the
    /// sketch. for the prefetcher, which asks about experts it may never need.
    #[must_use]
    pub fn contains(&self, key: ExpertKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// offer an expert to the cache after fetching it.
    ///
    /// it always enters, into the probationary window. it earns a place in the
    /// main region later, or it does not, and either way the caller does not
    /// have to care.
    pub fn admit(&mut self, key: ExpertKey, desc: ExpertDesc, value: T) -> Admission {
        self.stats.bytes_missed += desc.stored_bytes;

        if let Some(e) = self.entries.get_mut(&key) {
            // already resident, which happens when a bridged read delivers an
            // expert the caller had not noticed it already had
            e.value = value;
            return Admission::Admitted;
        }
        if desc.stored_bytes > self.config.capacity_bytes {
            self.stats.rejected_oversized += 1;
            return Admission::RejectedOversized;
        }

        self.version += 1;
        self.entries.insert(
            key,
            Entry {
                value,
                desc,
                residency: Residency::Quantized,
                region: Region::Window,
                score: 0.0,
                version: self.version,
            },
        );
        self.window.push_back(key);
        self.window_bytes += desc.stored_bytes;
        self.stats.admissions += 1;
        self.stats.bytes_admitted += desc.stored_bytes;
        self.observe_size(desc.stored_bytes);

        self.adapt_window();
        self.drain_window();
        while self.resident_bytes() > self.config.capacity_bytes && self.evict_main() {}

        Admission::Admitted
    }

    /// drop an expert if it is resident, returning its payload.
    pub fn remove(&mut self, key: ExpertKey) -> Option<T> {
        let e = self.entries.remove(&key)?;
        match e.region {
            Region::Window => {
                self.window_bytes -= e.resident_bytes();
                if let Some(pos) = self.window.iter().position(|&k| k == key) {
                    self.window.remove(pos);
                }
            }
            Region::Main => self.main_bytes -= e.resident_bytes(),
        }
        Some(e.value)
    }

    /// forget everything, including the frequency history.
    ///
    /// for a hard context switch where the previous domain's statistics are
    /// actively misleading rather than merely stale.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.heap.clear();
        self.window.clear();
        self.sketch.clear();
        self.window_bytes = 0;
        self.main_bytes = 0;
        self.clock = 0.0;
        self.window_budget = self.config.window_bytes();
        self.prev_hit_rate = -1.0;
        self.prev_hits = 0;
        self.prev_lookups = 0;
    }

    /// counters so far.
    #[must_use]
    pub const fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// reset the counters without disturbing what is resident.
    ///
    /// for measuring steady state after a warmup, which is the only honest way
    /// to report a hit rate.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// bytes currently held across both regions.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.window_bytes + self.main_bytes
    }

    /// the configured ceiling.
    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.config.capacity_bytes
    }

    /// number of resident experts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// whether nothing is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// which region an expert is in, if it is resident.
    #[must_use]
    pub fn region(&self, key: ExpertKey) -> Option<Region> {
        self.entries.get(&key).map(|e| e.region)
    }

    /// how an expert is currently held, if it is resident.
    #[must_use]
    pub fn residency(&self, key: ExpertKey) -> Option<Residency> {
        self.entries.get(&key).map(|e| e.residency)
    }

    /// the sketch's estimate of how often this expert has been used recently.
    #[must_use]
    pub fn estimated_frequency(&self, key: ExpertKey) -> u64 {
        self.sketch.estimate(key.packed())
    }

    /// every resident expert, in no particular order.
    pub fn keys(&self) -> impl Iterator<Item = ExpertKey> + '_ {
        self.entries.keys().copied()
    }

    /// keep the sketch's decay window matched to how many experts the cache can
    /// actually hold, which is not known until experts start arriving.
    fn observe_size(&mut self, stored_bytes: u64) {
        let n = self.stats.admissions as f64;
        self.mean_stored_bytes += (stored_bytes as f64 - self.mean_stored_bytes) / n;
        let items = (self.config.capacity_bytes as f64 / self.mean_stored_bytes.max(1.0)) as usize;
        self.sketch.set_cache_items(items);
    }

    /// push window victims into the main region until the window fits its
    /// budget.
    ///
    /// one entry always stays. if nothing can sit in the window then nothing can
    /// ever enter the cache at all, which is the deadlock this design exists to
    /// avoid.
    fn drain_window(&mut self) {
        while self.window_bytes > self.window_budget && self.window.len() > 1 {
            let Some(victim) = self.window.pop_front() else {
                break;
            };
            self.settle_window_victim(victim);
        }
    }

    /// hill climb the window size against the measured hit rate.
    ///
    /// the window and the main region want opposite things. a workload with
    /// strong short range recency, where everything accessed was seen moments
    /// ago, wants a large window, and that is exactly the ground on which plain
    /// lru is hard to beat. a workload with a stable skewed working set wants a
    /// large protected region instead. no fixed split is right for both, and a
    /// real conversation moves between them: bursty while a file is being
    /// edited, stable while a long answer is generated.
    ///
    /// so the split is not fixed. each interval the hit rate is compared with
    /// the previous interval, and if the last move helped it keeps going that
    /// way, otherwise it turns around. this is the adaptation from caffeine's
    /// window-tinylfu, and it is what stops the policy losing to lru on lru's
    /// own ground.
    fn adapt_window(&mut self) {
        if !self.config.adaptive_window {
            return;
        }
        let interval = self.adapt_interval();
        let lookups = self.stats.lookups();
        if lookups < self.prev_lookups + interval {
            return;
        }
        let d_lookups = lookups - self.prev_lookups;
        let d_hits = self.stats.hits - self.prev_hits;
        let hit_rate = d_hits as f64 / d_lookups as f64;

        if self.prev_hit_rate >= 0.0 && hit_rate < self.prev_hit_rate {
            self.adapt_grow = !self.adapt_grow;
        }
        let step = (self.config.capacity_bytes as f64 * self.config.adapt_step_fraction) as u64;
        let floor = self.config.window_bytes();
        let ceiling = ((self.config.capacity_bytes as f64 * self.config.max_window_fraction)
            as u64)
            .max(floor);
        self.window_budget = if self.adapt_grow {
            self.window_budget.saturating_add(step).min(ceiling)
        } else {
            self.window_budget.saturating_sub(step).max(floor)
        };

        self.prev_hit_rate = hit_rate;
        self.prev_hits = self.stats.hits;
        self.prev_lookups = lookups;
    }

    /// how many lookups to observe before moving the window again.
    ///
    /// tied to how many experts the cache holds, so that the hit rate being
    /// compared reflects the split rather than noise.
    fn adapt_interval(&self) -> u64 {
        let items = (self.config.capacity_bytes as f64 / self.mean_stored_bytes.max(1.0)) as u64;
        (10 * items).clamp(64, 100_000)
    }

    fn touch_window(&mut self, key: ExpertKey) {
        if let Some(pos) = self.window.iter().position(|&k| k == key) {
            self.window.remove(pos);
        }
        self.window.push_back(key);
    }

    fn score_for(&self, key: ExpertKey, stored_bytes: u64, resident_bytes: u64) -> f64 {
        let freq = self.sketch.estimate(key.packed()).max(1) as f64;
        let reload = self.config.cost.seconds_for(stored_bytes);
        self.clock + freq * reload / resident_bytes.max(1) as f64
    }

    fn rescore(&mut self, key: ExpertKey) {
        let Some(e) = self.entries.get(&key) else {
            return;
        };
        let (stored, resident) = (e.desc.stored_bytes, e.resident_bytes());
        let score = self.score_for(key, stored, resident);
        self.version += 1;
        let v = self.version;
        if let Some(e) = self.entries.get_mut(&key) {
            e.score = score;
            e.version = v;
        }
        self.heap.push(Victim {
            score,
            key,
            version: v,
        });
    }

    /// a window victim either earns a place in the main region or leaves.
    fn settle_window_victim(&mut self, key: ExpertKey) {
        let Some(e) = self.entries.get(&key) else {
            return;
        };
        let bytes = e.resident_bytes();
        let main_budget = self
            .config
            .capacity_bytes
            .saturating_sub(self.window_budget);

        while self.main_bytes + bytes > main_budget {
            let Some(loser) = self.peek_main_victim() else {
                break;
            };
            let contest_lost = self.config.tinylfu_admission
                && self.sketch.estimate(key.packed()) <= self.sketch.estimate(loser.packed());
            if contest_lost {
                self.drop_window_entry(key, bytes);
                self.stats.contention_losses += 1;
                return;
            }
            if !self.evict_main() {
                break;
            }
        }

        self.window_bytes -= bytes;
        self.main_bytes += bytes;
        self.version += 1;
        let v = self.version;
        let score = self.score_for(key, self.entries[&key].desc.stored_bytes, bytes);
        if let Some(e) = self.entries.get_mut(&key) {
            e.region = Region::Main;
            e.score = score;
            e.version = v;
        }
        self.heap.push(Victim {
            score,
            key,
            version: v,
        });
        self.stats.window_promotions += 1;
    }

    fn drop_window_entry(&mut self, key: ExpertKey, bytes: u64) {
        self.entries.remove(&key);
        self.window_bytes -= bytes;
        self.stats.evictions += 1;
        self.stats.bytes_evicted += bytes;
    }

    /// key of the main region entry that would be evicted next, discarding
    /// stale heap items as it goes.
    fn peek_main_victim(&mut self) -> Option<ExpertKey> {
        loop {
            let top = self.heap.peek()?;
            match self.entries.get(&top.key) {
                Some(e) if e.version == top.version && e.region == Region::Main => {
                    return Some(top.key);
                }
                _ => {
                    self.heap.pop();
                }
            }
        }
    }

    /// evict the lowest scoring main region entry. returns false if there was
    /// nothing to evict.
    fn evict_main(&mut self) -> bool {
        while let Some(top) = self.heap.pop() {
            let Some(e) = self.entries.get(&top.key) else {
                continue;
            };
            if e.version != top.version || e.region != Region::Main {
                continue;
            }
            let bytes = e.resident_bytes();
            // aging: the floor rises to whatever was just thrown out, so every
            // surviving entry's advantage erodes on its own
            self.clock = top.score;
            self.entries.remove(&top.key);
            self.main_bytes -= bytes;
            self.stats.evictions += 1;
            self.stats.bytes_evicted += bytes;
            return true;
        }
        false
    }

    /// hold a hot main region expert dequantised, trading capacity for compute
    /// at the point of use.
    ///
    /// deliberately conservative: promotion happens only into space that is
    /// already free. an expert that has to evict a peer in order to grow is not
    /// obviously worth it, and getting that wrong spends capacity on
    /// convenience rather than on coverage.
    fn maybe_promote(&mut self, key: ExpertKey) {
        let Some(e) = self.entries.get(&key) else {
            return;
        };
        if e.region != Region::Main || e.residency != Residency::Quantized {
            return;
        }
        let desc = e.desc;
        let extra = desc.dequantized_bytes.saturating_sub(desc.stored_bytes);
        if extra == 0 || self.sketch.estimate(key.packed()) < self.config.hot_promotion_freq {
            return;
        }
        if self.resident_bytes() + extra > self.config.capacity_bytes {
            return;
        }
        let score = self.score_for(key, desc.stored_bytes, desc.dequantized_bytes);
        self.version += 1;
        let v = self.version;
        if let Some(e) = self.entries.get_mut(&key) {
            e.residency = Residency::Dequantized;
            e.score = score;
            e.version = v;
        }
        self.main_bytes += extra;
        self.stats.promotions += 1;
        self.heap.push(Victim {
            score,
            key,
            version: v,
        });
    }
}
