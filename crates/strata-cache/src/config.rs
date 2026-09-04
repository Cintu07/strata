//! what the cache costs and how it is allowed to behave.

/// the device model the eviction policy prices reloads against.
///
/// this is not decoration. the eviction score divides reload cost by resident
/// bytes, and if reload cost were modelled as pure bandwidth then cost over
/// size would be the constant `1 / bandwidth` and the whole term would cancel,
/// leaving a plain popularity contest. it is the fixed per request latency that
/// makes the term mean something: a small expert is cheaper to keep per byte
/// than a large one, because the latency you avoid is the same either way.
///
/// measure these on the target device rather than copying a spec sheet.
/// consumer nvme misses its own sequential figure badly at shallow queue depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadCost {
    /// fixed cost of getting one request to first byte, in seconds.
    pub latency_s: f64,
    /// streaming rate once the request is moving, in bytes per second.
    pub bandwidth_bps: f64,
}

impl Default for ReadCost {
    /// a middling consumer gen4 nvme: 100 microseconds to first byte and
    /// 5 GB/s streaming. deliberately not the spec sheet number.
    fn default() -> Self {
        Self {
            latency_s: 100e-6,
            bandwidth_bps: 5.0e9,
        }
    }
}

impl ReadCost {
    /// seconds to fetch `bytes` from the cold tier.
    #[must_use]
    pub fn seconds_for(&self, bytes: u64) -> f64 {
        self.latency_s + bytes as f64 / self.bandwidth_bps
    }

    /// the expert size at which latency and transfer cost the same.
    ///
    /// below it a read is dominated by the round trip, which is the regime
    /// where holding many small experts beats holding a few large ones, and the
    /// regime where coalescing neighbouring reads pays best.
    #[must_use]
    pub fn latency_equivalent_bytes(&self) -> f64 {
        self.latency_s * self.bandwidth_bps
    }
}

/// how the cache is allowed to spend its ram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheConfig {
    /// hard ceiling on resident expert bytes. hot weights, the kv cache, and
    /// prefetch staging buffers are budgeted separately and are not counted
    /// here, so this number is smaller than the ram you have.
    pub capacity_bytes: u64,
    /// device model used to price reloads.
    pub cost: ReadCost,
    /// rough number of distinct expert-layer pairs in the model.
    ///
    /// sizes the frequency sketch and the window over which its counters decay.
    /// it does not have to be exact; the sketch degrades gracefully, and being
    /// out by a factor of two costs a little accuracy in the long tail and
    /// nothing at all in the head.
    pub expected_experts: usize,
    /// fraction of the capacity held as a probationary window.
    ///
    /// new experts always land here first and are only asked to justify
    /// themselves when they fall out of it. one percent is the figure the
    /// tinylfu line of work converged on and it is a reasonable default, but it
    /// is the knob to turn if a workload is dominated by short lived bursts:
    /// a larger window absorbs bursts, a smaller one protects the stable set.
    pub window_fraction: f64,
    /// whether a window victim has to outrank the expert it would displace.
    ///
    /// turning this off is the ablation that separates the effect of admission
    /// from the effect of the eviction score. a comparison that changes two
    /// things at once explains nothing.
    pub tinylfu_admission: bool,
    /// whether the window size hill climbs against the measured hit rate.
    ///
    /// **off by default, because it was measured and it did not pay.** the
    /// mechanism is caffeine's, and the reasoning for it is sound: no fixed
    /// split suits both a bursty workload and a stable one. but on the three
    /// synthetic workloads in `tests/measure.rs` it lost ground almost
    /// everywhere it moved, worst on the skewed single layer case, where a
    /// small cache cannot spare a growing probationary window at all.
    ///
    /// tuning an adaptive controller against synthetic workloads is fitting
    /// noise, so it stays off until there are real router traces from m0 to
    /// tune it against. the code is here and the flag turns it on.
    pub adaptive_window: bool,
    /// how far the window moves at each adaptation, as a fraction of capacity.
    pub adapt_step_fraction: f64,
    /// ceiling on the window, as a fraction of capacity.
    ///
    /// at the ceiling the policy is close to plain lru, which is the right
    /// answer for a workload with no reusable frequency structure and the wrong
    /// answer for every other one, so it does not go all the way to one.
    pub max_window_fraction: f64,
    /// estimated access count at which an expert is worth holding dequantised.
    ///
    /// counts saturate at fifteen, so anything above that disables segmented
    /// residency entirely.
    pub hot_promotion_freq: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity_bytes: 4 << 30,
            cost: ReadCost::default(),
            expected_experts: 4096,
            window_fraction: 0.01,
            tinylfu_admission: true,
            adaptive_window: false,
            adapt_step_fraction: 0.05,
            max_window_fraction: 0.8,
            hot_promotion_freq: 8,
        }
    }
}

impl CacheConfig {
    /// a cache of the given size with everything else left at its default.
    #[must_use]
    pub fn with_capacity(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            ..Self::default()
        }
    }

    /// starting size of the probationary window, and the floor it may not adapt
    /// below.
    ///
    /// never zero: if nothing can enter the window then nothing can ever enter
    /// the cache, which is the deadlock this design exists to avoid.
    #[must_use]
    pub fn window_bytes(&self) -> u64 {
        let w = (self.capacity_bytes as f64 * self.window_fraction) as u64;
        w.max(1)
    }
}
