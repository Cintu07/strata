//! convert a safetensors checkpoint into a strata layout file.
//!
//! ```text
//! cargo run --release -p strata-convert --bin convert -- \
//!     m0/models/granite-1b-a400m/model.safetensors granite.strata
//! ```
//!
//! pass `--plan` to print what it would do and write nothing, which is the
//! cheap way to find out whether a new model family is recognised.

use std::process::ExitCode;
use strata_convert::{SafeTensors, plan};
use strata_format::{CoactivationEdge, LayoutReader, RouteTrace};
use strata_layout::plan_layout;

/// co-activation edges below this fraction of a layer's tokens are noise, and
/// ordering by them would fit the corpus rather than the model.
const MIN_EDGE_WEIGHT: f32 = 0.02;

/// what the command line asked for.
struct Args {
    source: String,
    out: String,
    plan_only: bool,
    verify: bool,
    order: Option<String>,
}

/// parse arguments, or `None` if there is nothing to do.
fn parse_args() -> Option<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    let order = argv
        .iter()
        .position(|a| a == "--order")
        .and_then(|i| argv.get(i + 1))
        .cloned();

    // --order takes a value, so its argument is not a positional
    let mut positional = Vec::new();
    let mut skip_next = false;
    for arg in &argv {
        if std::mem::replace(&mut skip_next, false) {
            continue;
        }
        if arg == "--order" {
            skip_next = true;
        } else if !arg.starts_with("--") {
            positional.push(arg.clone());
        }
    }

    let source = positional.first()?.clone();
    Some(Args {
        source,
        out: positional
            .get(1)
            .cloned()
            .unwrap_or_else(|| "model.strata".to_string()),
        plan_only: argv.iter().any(|a| a == "--plan"),
        verify: argv.iter().any(|a| a == "--verify"),
        order,
    })
}

fn main() -> ExitCode {
    let Some(args) = parse_args() else {
        eprintln!(
            "usage: convert <model.safetensors> [out.strata] [options]\n\n\
             --plan     print the conversion plan and write nothing\n\
             --verify   after writing, compare every expert against the source\n\
             --order T  lay the file out by the routing measured in trace T"
        );
        return ExitCode::from(2);
    };
    let (source_path, out_path) = (args.source.as_str(), args.out.as_str());
    let (plan_only, verify, order_path) = (args.plan_only, args.verify, args.order.as_deref());

    let mut source = match SafeTensors::open(source_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot open {source_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut plan = match plan::plan(&source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot plan {source_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let payload: u64 = plan.experts.iter().map(|e| e.payload_len).sum();
    println!(
        "{} layers x {} experts = {} expert-layer pairs, {:?}",
        plan.layers,
        plan.experts_per_layer,
        plan.experts.len(),
        plan.precision
    );
    println!("projections per expert: {}", plan.projections.join(", "));
    println!(
        "payload {:.2} GiB, {:.2} MiB per expert",
        payload as f64 / (1u64 << 30) as f64,
        plan.experts.first().map_or(0.0, |e| e.payload_len as f64) / (1u64 << 20) as f64
    );

    let mut edges = Vec::new();
    if let Some(route) = order_path {
        match apply_ordering(&mut plan, route) {
            Ok(e) => edges = e,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }

    if plan_only {
        println!("\n--plan given, nothing written");
        return ExitCode::SUCCESS;
    }

    let started = std::time::Instant::now();
    let report =
        match strata_convert::convert_with_edges(&mut source, &plan, out_path, source_path, &edges)
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("conversion failed: {e}");
                return ExitCode::FAILURE;
            }
        };

    let secs = started.elapsed().as_secs_f64();
    let gib = report.bytes_written as f64 / (1u64 << 30) as f64;
    println!(
        "\nwrote {} experts, {gib:.2} GiB to {out_path} in {secs:.1}s ({:.2} GiB/s)",
        report.experts,
        gib / secs
    );

    if !verify {
        println!("pass --verify to check every expert against the source");
        return ExitCode::SUCCESS;
    }

    match verify_all(&mut source, &plan, out_path) {
        Ok(checked) => {
            println!("verified {checked} experts byte for byte against the source");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("VERIFICATION FAILED: {e}");
            ExitCode::FAILURE
        }
    }
}

/// reorder the plan by measured routing, and return the graph to store with it.
///
/// # why the shape check is fatal rather than a warning
///
/// an ordering derived from a different model's routing is not a worse
/// ordering, it is a meaningless one, and it would produce a layout file that
/// looks profile guided and is arbitrary. refusing is the only honest option.
fn apply_ordering(
    plan: &mut strata_convert::ModelPlan,
    route: &str,
) -> Result<Vec<CoactivationEdge>, String> {
    let trace = RouteTrace::load(route).map_err(|e| format!("cannot read {route}: {e}"))?;

    if trace.n_layers != plan.layers as usize || trace.n_experts != plan.experts_per_layer as usize
    {
        return Err(format!(
            "trace is {} layers x {} experts, model is {} x {}. ordering a model \
             by another model's routing would be worse than not ordering it.",
            trace.n_layers, trace.n_experts, plan.layers, plan.experts_per_layer
        ));
    }

    let profile = strata_convert::profile_from_trace(&trace, plan);
    let order = plan_layout(&profile, MIN_EDGE_WEIGHT);
    strata_convert::reorder(plan, &order);
    let edges = profile.edges(MIN_EDGE_WEIGHT);

    println!(
        "ordered by {} tokens of observed routing, {} co-activation edges kept",
        trace.n_tokens,
        edges.len()
    );
    Ok(edges)
}

/// re-read every expert from both files and compare.
///
/// this is the only check that means anything about a converter. a wrong offset
/// produces a file of exactly the right length full of plausible weights, and
/// crc32 in the index only proves the layout file is internally consistent with
/// whatever it was given, not that what it was given was correct.
fn verify_all(
    source: &mut SafeTensors,
    plan: &strata_convert::ModelPlan,
    out_path: &str,
) -> Result<usize, String> {
    let reader = LayoutReader::open(out_path).map_err(|e| format!("open layout: {e}"))?;
    let mut want = Vec::new();

    for expert in &plan.experts {
        let got = reader
            .read_expert(expert.key)
            .map_err(|e| format!("read {}: {e}", expert.key))?;

        want.clear();
        want.resize(expert.payload_len as usize, 0);
        let mut at = 0usize;
        for part in &expert.parts {
            let end = at + part.len as usize;
            source
                .read_into(&part.tensor, part.offset, &mut want[at..end])
                .map_err(|e| format!("re-read source for {}: {e}", expert.key))?;
            at = end;
        }

        if got != want {
            let first = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            return Err(format!(
                "{} differs from the source at byte {first} of {}",
                expert.key, expert.payload_len
            ));
        }
    }
    Ok(plan.experts.len())
}
