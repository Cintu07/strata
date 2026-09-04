//! measure what the device actually does, against queue depth and request size.
//!
//! two numbers in the prd depend entirely on hardware and cannot be reasoned
//! about from a spec sheet:
//!
//! - **m2's gate**: sequential read bandwidth above 80 percent of device spec
//! - **the storage risk**: consumer nvme random reads at shallow queue depth are
//!   far worse than the sheet claims, and the whole layout design exists to
//!   avoid that regime
//!
//! this binary answers both. the table it prints is the justification for the
//! read planner's coalescing defaults and for `StorageConfig::queue_depth`, and
//! it should be rerun on any machine those are being tuned for.
//!
//! ```text
//! cargo run --release -p strata-io --bin bandwidth -- /path/on/the/target/device
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use strata_io::{Completion, ReadOp, StorageConfig};

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;

/// how long to run each configuration. long enough to get past a burst of cache
/// hits, short enough that the whole sweep is a couple of minutes.
const RUN_FOR: Duration = Duration::from_millis(1500);

struct Sample {
    pattern: &'static str,
    block: usize,
    queue_depth: usize,
    bytes: u64,
    ops: u64,
    elapsed: Duration,
}

impl Sample {
    fn gb_per_s(&self) -> f64 {
        self.bytes as f64 / self.elapsed.as_secs_f64() / 1e9
    }
    fn iops(&self) -> f64 {
        self.ops as f64 / self.elapsed.as_secs_f64()
    }
    /// mean time each request spent outstanding, which is the number the
    /// prefetch horizon has to be long enough to cover.
    fn mean_latency_us(&self) -> f64 {
        if self.ops == 0 {
            return 0.0;
        }
        self.elapsed.as_secs_f64() * 1e6 * self.queue_depth as f64 / self.ops as f64
    }
}

/// xorshift, so a run is reproducible and two runs are comparable.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn ensure_file(path: &PathBuf, bytes: u64) -> std::io::Result<()> {
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() >= bytes {
            return Ok(());
        }
    }
    eprintln!(
        "creating a {} GiB test file at {}",
        bytes / (1 << 30),
        path.display()
    );
    let mut f = std::fs::File::create(path)?;
    let mut block = vec![0u8; 4 * MIB];
    let mut rng = Rng(0x0000_5EED);
    for (i, b) in block.iter_mut().enumerate() {
        *b = (rng.next() ^ i as u64) as u8;
    }
    let mut written = 0u64;
    while written < bytes {
        let n = block.len().min((bytes - written) as usize);
        f.write_all(&block[..n])?;
        written += n as u64;
    }
    f.sync_all()?;
    Ok(())
}

/// keep `queue_depth` reads outstanding for `RUN_FOR` and report what landed.
fn run(
    path: &PathBuf,
    file_bytes: u64,
    pattern: &'static str,
    block: usize,
    queue_depth: usize,
) -> std::io::Result<Sample> {
    let config = StorageConfig {
        queue_depth,
        slot_bytes: block,
        alignment: 4096,
        direct: true,
    };
    let (mut storage, _) = strata_io::open_best(path, config)?;

    let blocks = file_bytes / block as u64;
    let mut rng = Rng(0x00C0_FFEE);
    let mut cursor = 0u64;
    let next_offset = move |rng: &mut Rng, cursor: &mut u64| -> u64 {
        match pattern {
            "sequential" => {
                let off = *cursor % blocks;
                *cursor += 1;
                off * block as u64
            }
            _ => (rng.next() % blocks) * block as u64,
        }
    };

    let mut done = Vec::new();
    let mut bytes = 0u64;
    let mut ops = 0u64;
    let mut id = 0u64;
    let start = Instant::now();

    // prime the queue
    while storage.available() > 0 {
        let off = next_offset(&mut rng, &mut cursor);
        if storage
            .submit(ReadOp {
                id,
                offset: off,
                len: block,
            })?
            .is_none()
        {
            break;
        }
        id += 1;
    }
    storage.flush()?;

    while start.elapsed() < RUN_FOR {
        storage.wait(1, &mut done)?;
        let reaped: Vec<Completion> = std::mem::take(&mut done);
        for c in reaped {
            if c.result.is_ok() {
                bytes += block as u64;
                ops += 1;
            }
            storage.release(c.slot);
        }
        while storage.available() > 0 {
            let off = next_offset(&mut rng, &mut cursor);
            if storage
                .submit(ReadOp {
                    id,
                    offset: off,
                    len: block,
                })?
                .is_none()
            {
                break;
            }
            id += 1;
        }
        storage.flush()?;
    }
    let elapsed = start.elapsed();

    // drain, so the next configuration starts from a quiet device
    while storage.in_flight() > 0 {
        storage.wait(1, &mut done)?;
        for c in done.drain(..) {
            storage.release(c.slot);
        }
    }

    Ok(Sample {
        pattern,
        block,
        queue_depth,
        bytes,
        ops,
        elapsed,
    })
}

fn human(bytes: usize) -> String {
    if bytes >= MIB {
        format!("{}M", bytes / MIB)
    } else {
        format!("{}K", bytes / KIB)
    }
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: bandwidth <path-on-target-device> [size-gib]");
        std::process::exit(2);
    }));
    let gib: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let file_bytes = gib << 30;

    ensure_file(&path, file_bytes)?;

    let (probe, backend) = strata_io::open_best(&path, StorageConfig::default())?;
    drop(probe);

    println!();
    println!("strata storage bandwidth");
    println!("  path      {}", path.display());
    println!("  size      {gib} GiB");
    println!("  backend   {backend}");
    println!("  direct io yes, so these are device numbers and not page cache numbers");
    println!();
    println!(
        "  {:<11} {:>6} {:>4} {:>9} {:>11} {:>10}",
        "pattern", "block", "qd", "GB/s", "IOPS", "lat us"
    );
    println!("  {}", "-".repeat(56));

    let mut sequential_peak = 0.0f64;
    let mut qd1_random_4k = 0.0f64;
    let mut qd_high_random_4k = 0.0f64;

    for (pattern, block) in [
        ("sequential", MIB),
        ("random", 4 * KIB),
        ("random", 64 * KIB),
        ("random", MIB),
    ] {
        for qd in [1usize, 4, 16, 64, 128] {
            let s = run(&path, file_bytes, pattern, block, qd)?;
            println!(
                "  {:<11} {:>6} {:>4} {:>9.3} {:>11.0} {:>10.1}",
                s.pattern,
                human(s.block),
                s.queue_depth,
                s.gb_per_s(),
                s.iops(),
                s.mean_latency_us()
            );
            if pattern == "sequential" {
                sequential_peak = sequential_peak.max(s.gb_per_s());
            }
            if pattern == "random" && block == 4 * KIB {
                if qd == 1 {
                    qd1_random_4k = s.gb_per_s();
                }
                qd_high_random_4k = qd_high_random_4k.max(s.gb_per_s());
            }
        }
        println!();
    }

    println!("what this says");
    println!("  peak sequential          {sequential_peak:.2} GB/s");
    println!("  random 4k at qd1         {qd1_random_4k:.3} GB/s");
    println!("  random 4k at best qd     {qd_high_random_4k:.3} GB/s");
    if qd1_random_4k > 0.0 {
        println!(
            "  queue depth is worth     {:.0}x on random 4k",
            qd_high_random_4k / qd1_random_4k
        );
    }
    if sequential_peak > 0.0 {
        println!(
            "  sequential is worth      {:.0}x over random 4k at qd1",
            sequential_peak / qd1_random_4k.max(1e-9)
        );
    }
    println!();
    println!("  the second and third numbers are the whole argument for the layout file:");
    println!("  the same bytes cost wildly different amounts depending on how they are asked for.");
    println!("  set StorageConfig::queue_depth from the qd column where GB/s stops improving.");
    println!();
    println!("  compare the random 1M rows against the sequential rows. if they are close,");
    println!("  then on this device request *size* matters more than adjacency, and the");
    println!("  layout should be optimising for large reads rather than for contiguity.");
    println!();
    println!("  caveat: if this ran inside a vm, the filesystem sits on a virtual disk on a");
    println!("  host filesystem, and both layers are in the path. treat the shape of the");
    println!("  curve as real and the absolute peak as a lower bound on the bare device.");

    Ok(())
}
