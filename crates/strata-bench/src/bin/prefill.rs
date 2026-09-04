//! what the three mechanisms are worth together, measured against real io.
//!
//! this builds a synthetic expert file on the target device and runs one prefill
//! layer through it four ways, adding one mechanism at a time:
//!
//! 1. **token-major, one read per token-slot.** what a naive implementation
//!    does. every token independently demands its experts.
//! 2. **expert-major.** o3. invert to expert-to-token, so each expert is read
//!    once per layer instead of once per token that wanted it.
//! 3. **expert-major in disk order, coalesced.** add the read planner, so
//!    neighbouring experts merge into single large transfers.
//! 4. **plus a warm expert cache.** the steady state, where a large part of the
//!    working set is already resident.
//!
//! the weights are synthetic, so this measures the io path and the scheduling
//! and not a model. that is deliberate: those are the parts that exist, and
//! putting a real model behind it would change the number without changing what
//! is being tested.
//!
//! ```text
//! cargo run --release -p strata-bench --bin prefill -- /path/on/the/target/device
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};
use strata_cache::{CacheConfig, ExpertCache, ExpertDesc};
use strata_format::{ExpertKey, LayoutReader, LayoutWriter, PlanOptions, Precision};
use strata_io::{ReadOp, StorageConfig};
use strata_prefill::{LayerRouting, schedule_layer};

const EXPERT_BYTES: usize = 2 * 1024 * 1024;
const N_EXPERTS: u32 = 128;
const N_LAYERS: u32 = 4;
const N_TOKENS: usize = 1024;
const TOP_K: usize = 8;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// skewed towards low indices, which is what a real router does.
    fn skewed(&mut self, n: u64) -> u64 {
        (0..3).map(|_| self.below(n)).min().unwrap_or(0)
    }
}

struct Result {
    name: &'static str,
    reads: usize,
    bytes: u64,
    elapsed: Duration,
    /// what fraction of the full prefill this row actually performed.
    ///
    /// token-major is far too slow to run in full for a benchmark, so it runs a
    /// prefix. carrying the fraction here means the table can never quietly
    /// compare a slice of one strategy against the whole of another.
    fraction: f64,
}

impl Result {
    fn gb_per_s(&self) -> f64 {
        self.bytes as f64 / self.elapsed.as_secs_f64() / 1e9
    }

    /// the row scaled to a whole prefill.
    fn full_bytes(&self) -> u64 {
        (self.bytes as f64 / self.fraction) as u64
    }

    fn full_ms(&self) -> f64 {
        self.elapsed.as_secs_f64() * 1000.0 / self.fraction
    }

    fn is_extrapolated(&self) -> bool {
        self.fraction < 1.0
    }
}

fn build_file(path: &PathBuf) {
    if std::fs::metadata(path).is_ok_and(|m| m.len() > 0) {
        return;
    }
    eprintln!(
        "building a {} MiB expert file at {}",
        (N_EXPERTS as usize * N_LAYERS as usize * EXPERT_BYTES) >> 20,
        path.display()
    );
    let mut w = LayoutWriter::create(path, "bench-moe").expect("create layout file");
    let payload = vec![0xA5u8; EXPERT_BYTES];
    for layer in 0..N_LAYERS {
        for e in 0..N_EXPERTS {
            w.push_expert(ExpertKey::new(layer, e), Precision::Q4, &payload)
                .expect("push expert");
        }
    }
    w.finish().expect("finish layout file");
}

fn routing(layer: u32, seed: u64) -> LayerRouting {
    let mut rng = Rng(seed | 1);
    let mut experts = Vec::with_capacity(N_TOKENS * TOP_K);
    let mut weights = Vec::with_capacity(N_TOKENS * TOP_K);
    for _ in 0..N_TOKENS {
        let mut chosen: Vec<u32> = Vec::new();
        while chosen.len() < TOP_K {
            let e = rng.skewed(u64::from(N_EXPERTS)) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        for e in chosen {
            experts.push(e);
            weights.push(0.125);
        }
    }
    LayerRouting::new(layer, TOP_K, experts, weights)
}

/// issue a list of transfers through `io_uring`, keeping the queue full.
fn execute(
    path: &PathBuf,
    transfers: &[(u64, usize)],
    queue_depth: usize,
) -> std::io::Result<(u64, Duration)> {
    let slot_bytes = transfers.iter().map(|&(_, len)| len).max().unwrap_or(4096);
    let config = StorageConfig {
        queue_depth,
        slot_bytes: slot_bytes.next_multiple_of(4096),
        alignment: 4096,
        direct: true,
    };
    let (mut storage, _) = strata_io::open_best(path, config)?;

    let start = Instant::now();
    let mut done = Vec::new();
    let mut bytes = 0u64;
    let mut next = 0usize;
    let mut queued = 0usize;

    while next < transfers.len() || queued > 0 {
        while next < transfers.len() {
            let (offset, len) = transfers[next];
            match storage.submit(ReadOp {
                id: next as u64,
                offset,
                len,
            })? {
                Some(_) => {
                    queued += 1;
                    next += 1;
                }
                None => break,
            }
        }
        storage.flush()?;
        if queued == 0 {
            continue;
        }
        storage.wait(1, &mut done)?;
        for c in done.drain(..) {
            if let Ok(n) = c.result {
                bytes += n as u64;
            }
            storage.release(c.slot);
            queued -= 1;
        }
    }
    Ok((bytes, start.elapsed()))
}

#[allow(clippy::too_many_lines)] // a benchmark is a script, and splitting it
// into helpers would hide the sequence of stages that is the whole point
fn main() -> std::io::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: prefill <path-on-target-device>");
        std::process::exit(2);
    }));
    build_file(&path);

    let reader = LayoutReader::open(&path).expect("open layout file");
    let data_off = reader.header().data_off;
    let mut results = Vec::new();

    // ---- 1. token major: one read per token-slot, in whatever order tokens ask
    {
        let mut transfers = Vec::new();
        for layer in 0..N_LAYERS {
            let r = routing(layer, u64::from(layer) + 7);
            for token in 0..r.n_tokens() {
                for slot in 0..r.top_k() {
                    let key = ExpertKey::new(layer, r.expert_at(token, slot));
                    let e = reader.entry(key).expect("expert in file");
                    transfers.push((e.file_offset(data_off), e.padded_len() as usize));
                }
            }
        }
        // running this in full takes minutes, so measure a prefix and record
        // what fraction it was, rather than comparing a slice of this against
        // the whole of everything else
        let total = transfers.len();
        transfers.truncate(4096);
        let fraction = transfers.len() as f64 / total as f64;
        let (bytes, elapsed) = execute(&path, &transfers, 64)?;
        results.push(Result {
            name: "token-major",
            reads: total,
            bytes,
            elapsed,
            fraction,
        });
    }

    // ---- 2. expert major: one read per expert per layer
    let mut expert_major_transfers = Vec::new();
    for layer in 0..N_LAYERS {
        let r = routing(layer, u64::from(layer) + 7);
        let schedule = schedule_layer(&r, &reader);
        for batch in &schedule.batches {
            let e = reader.entry(batch.key).expect("expert in file");
            expert_major_transfers.push((e.file_offset(data_off), e.padded_len() as usize));
        }
    }
    {
        let (bytes, elapsed) = execute(&path, &expert_major_transfers, 64)?;
        results.push(Result {
            name: "expert-major",
            reads: expert_major_transfers.len(),
            bytes,
            elapsed,
            fraction: 1.0,
        });
    }

    // ---- 3. expert major, coalesced by the read planner
    let mut coalesced = Vec::new();
    let mut planned_reads = 0usize;
    for layer in 0..N_LAYERS {
        let r = routing(layer, u64::from(layer) + 7);
        let schedule = schedule_layer(&r, &reader);
        let plan = schedule
            .read_plan(&reader, PlanOptions::default())
            .expect("plan reads");
        planned_reads += plan.requests.len();
        for req in &plan.requests {
            coalesced.push((req.file_offset, req.len as usize));
        }
    }
    {
        let (bytes, elapsed) = execute(&path, &coalesced, 64)?;
        results.push(Result {
            name: "+ coalesced",
            reads: planned_reads,
            bytes,
            elapsed,
            fraction: 1.0,
        });
    }

    // ---- 4. plus a warm cache, so only the misses reach the device
    let cache_bytes = (u64::from(N_EXPERTS) * u64::from(N_LAYERS) * EXPERT_BYTES as u64) / 4;
    let mut cache: ExpertCache<()> = ExpertCache::new(CacheConfig {
        expected_experts: (N_EXPERTS * N_LAYERS) as usize,
        ..CacheConfig::with_capacity(cache_bytes)
    });
    let desc = ExpertDesc::plain(EXPERT_BYTES as u64);

    // warm on the first pass, measure the second. the counters are reset in
    // between, because a hit rate that includes its own cold start is not a
    // steady state number and reporting one would flatter the cache.
    for pass in 0..2 {
        if pass == 1 {
            cache.reset_stats();
        }
        let mut misses = Vec::new();
        for layer in 0..N_LAYERS {
            let r = routing(layer, u64::from(layer) + 7);
            let schedule = schedule_layer(&r, &reader);
            for batch in &schedule.batches {
                if cache.get(batch.key).is_none() {
                    let e = reader.entry(batch.key).expect("expert in file");
                    misses.push((e.file_offset(data_off), e.padded_len() as usize));
                    cache.admit(batch.key, desc, ());
                }
            }
        }
        if pass == 1 {
            let (bytes, elapsed) = execute(&path, &misses, 64)?;
            results.push(Result {
                name: "+ warm cache",
                reads: misses.len(),
                bytes,
                elapsed,
                fraction: 1.0,
            });
        }
    }
    let hit_rate = cache.stats().hit_rate();

    println!();
    println!("strata prefill, {N_TOKENS} tokens x {N_LAYERS} layers, top-{TOP_K} of {N_EXPERTS}");
    println!("  expert size   {} MiB", EXPERT_BYTES >> 20);
    println!("  file          {}", path.display());
    println!();
    println!(
        "  {:<14} {:>8} {:>12} {:>9} {:>10}",
        "stage", "reads", "bytes read", "GB/s", "time ms"
    );
    println!("  {}", "-".repeat(68));
    for r in &results {
        // every row is stated for a whole prefill, with extrapolated ones marked
        println!(
            "  {:<14} {:>8} {:>11} M {:>9.2} {:>10.1}  {}",
            r.name,
            r.reads,
            r.full_bytes() >> 20,
            r.gb_per_s(),
            r.full_ms(),
            if r.is_extrapolated() {
                format!("extrapolated from {:.0}%", r.fraction * 100.0)
            } else {
                "measured in full".to_string()
            }
        );
    }

    let token_major_reads = N_TOKENS * TOP_K * N_LAYERS as usize;
    let naive = &results[0];
    let reordered = &results[1];
    let planned = &results[2];

    println!();
    println!("what this says");
    println!(
        "  token-major issues {token_major_reads} reads for this block and moves {} MiB.",
        naive.full_bytes() >> 20
    );
    println!(
        "  expert-major issues {} and moves {} MiB, which is g4:",
        expert_major_transfers.len(),
        reordered.full_bytes() >> 20
    );
    println!(
        "    {:.0}x fewer reads, {:.0}x fewer bytes, {:.1}x less time.",
        token_major_reads as f64 / expert_major_transfers.len() as f64,
        naive.full_bytes() as f64 / reordered.full_bytes().max(1) as f64,
        naive.full_ms() / reordered.full_ms().max(0.001)
    );
    println!("  coalescing then merges those into {planned_reads} transfers.");
    println!();
    println!(
        "  note the coalesced row: {:.0} ms against {:.0} ms, for {:.0}x fewer requests.",
        planned.full_ms(),
        reordered.full_ms(),
        expert_major_transfers.len() as f64 / planned_reads.max(1) as f64
    );
    println!(
        "  at {} MiB per expert the reads were already large enough to saturate the",
        EXPERT_BYTES >> 20
    );
    println!("  device, so merging them buys close to nothing. coalescing earns its place");
    println!("  when experts are small, which is where request latency dominates. run the");
    println!("  bandwidth binary and compare the random 64K and random 1M rows to see the");
    println!("  size at which that stops being true on a given device.");
    println!();
    println!(
        "  the warm cache removes a further {:.0}% of the remaining reads.",
        hit_rate * 100.0
    );
    println!();
    println!("  the weights here are synthetic, so this measures the io path and the");
    println!("  scheduling, not a model. those are the parts that exist.");

    Ok(())
}
