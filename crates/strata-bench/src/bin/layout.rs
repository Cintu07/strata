//! does laying the file out by measured co-activation actually save reads?
//!
//! ```text
//! cargo run --release -p strata-bench --bin layout -- \
//!     baseline.strata ordered.strata granite.route
//! ```
//!
//! # what it measures
//!
//! it replays the real want-sets: for every token, at every layer, the set of
//! experts that token actually routed to. each of those goes through the read
//! planner against both layout files, and the totals are request count, bytes
//! transferred, and overfetch.
//!
//! request count is the number to look at. decision 0009 measured that once a
//! read is a megabyte or more the request count stops mattering to throughput,
//! so a layout that saves requests on 3 mib experts may save nothing at all in
//! time. that is a real possibility and this prints the numbers either way
//! rather than only the ones that flatter the layout.

use std::process::ExitCode;
use strata_format::{ExpertKey, LayoutReader, PlanOptions, RouteTrace};

/// one layout file's totals over the whole trace.
struct Totals {
    requests: u64,
    transferred: u64,
    wanted: u64,
}

impl Totals {
    fn overfetch(&self) -> f64 {
        if self.wanted == 0 {
            1.0
        } else {
            self.transferred as f64 / self.wanted as f64
        }
    }
}

fn measure(reader: &LayoutReader, trace: &RouteTrace, opts: PlanOptions) -> Result<Totals, String> {
    let mut totals = Totals {
        requests: 0,
        transferred: 0,
        wanted: 0,
    };
    let mut want: Vec<ExpertKey> = Vec::with_capacity(trace.top_k);

    for token in 0..trace.n_tokens {
        for layer in 0..trace.n_layers {
            // the want-set is one layer of one token: what the router asked for
            // before anything is cached. duplicates are removed because a
            // planner is handed a set, not a multiset.
            want.clear();
            want.extend_from_slice(trace.selection(token, layer));
            want.sort_unstable();
            want.dedup();

            let plan = reader
                .plan_reads(&want, opts)
                .map_err(|e| format!("planning token {token} layer {layer}: {e}"))?;
            totals.requests += plan.requests.len() as u64;
            totals.transferred += plan.transferred_bytes;
            totals.wanted += plan.wanted_bytes;
        }
    }
    Ok(totals)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: layout <baseline.strata> <ordered.strata> <trace.route>");
        return ExitCode::from(2);
    }

    let trace = match RouteTrace::load(&args[2]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[2]);
            return ExitCode::FAILURE;
        }
    };

    let mut readers = Vec::new();
    for path in &args[0..2] {
        match LayoutReader::open(path) {
            Ok(r) => readers.push((path.clone(), r)),
            Err(e) => {
                eprintln!("cannot open {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    println!(
        "\n{} tokens x {} layers x top-{} of {} experts",
        trace.n_tokens, trace.n_layers, trace.top_k, trace.n_experts
    );
    println!(
        "{} want-sets planned per layout\n",
        trace.n_tokens * trace.n_layers
    );

    // three planner settings, because the answer depends on what the planner is
    // allowed to do and reporting one setting would be picking the flattering one
    let settings: [(&str, PlanOptions); 3] = [
        (
            "per expert, no coalescing",
            PlanOptions {
                coalesce: false,
                ..PlanOptions::default()
            },
        ),
        ("adjacent only", PlanOptions::no_overfetch()),
        ("default, 1 MiB gap", PlanOptions::default()),
    ];

    for (label, opts) in settings {
        println!("== {label}");
        println!(
            "  {:<28} {:>12} {:>14} {:>11}",
            "layout", "requests", "GiB moved", "overfetch"
        );
        let mut first: Option<u64> = None;
        for (path, reader) in &readers {
            let t = match measure(reader, &trace, opts) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            };
            let name = path.rsplit('/').next().unwrap_or(path);
            let delta = match first {
                None => {
                    first = Some(t.requests);
                    String::new()
                }
                Some(base) => {
                    let change = (t.requests as f64 - base as f64) / base as f64 * 100.0;
                    format!("  {change:+.1}% requests")
                }
            };
            println!(
                "  {:<28} {:>12} {:>14.2} {:>11.3}{}",
                name,
                t.requests,
                t.transferred as f64 / (1u64 << 30) as f64,
                t.overfetch(),
                delta
            );
        }
        println!();
    }

    ExitCode::SUCCESS
}
