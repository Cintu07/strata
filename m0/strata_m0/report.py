"""the five plots and the go/no-go verdict.

m0 is not a warm-up, it is the project's falsification test. this module is
where that test is actually adjudicated: it runs every analysis, draws the
figures, and writes a verdict that says in plain terms whether the design's
assumptions survived contact with a real model.

a negative result here is a real outcome and the report is written to say so
clearly rather than to bury it. three weeks and a clean set of numbers nobody
has published is a contribution. a year spent on a false premise is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from . import analysis, cache_sim, predict  # noqa: E402
from .trace import RouterTrace  # noqa: E402

PASS, MARGINAL, FAIL = "pass", "marginal", "fail"


@dataclass
class Check:
    """one go/no-go criterion with the number that decided it."""

    name: str
    verdict: str
    value: float
    threshold: float
    matters_because: str

    def line(self) -> str:
        return (
            f"| {self.name} | **{self.verdict}** | {self.value:.3f} | "
            f"{self.threshold:.3f} | {self.matters_because} |"
        )


def _verdict(value: float, threshold: float, margin: float = 0.8) -> str:
    if value >= threshold:
        return PASS
    if value >= threshold * margin:
        return MARGINAL
    return FAIL


# ------------------------------------------------------------------- plots


def plot_reuse(trace: RouterTrace, out: Path) -> Path:
    reuse = analysis.reuse_across_tokens(trace)
    curve = analysis.reuse_curve(trace, max_distance=16)
    chance = trace.top_k / trace.n_experts

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.bar(np.arange(trace.n_layers), reuse.per_layer, color="#3c6e91")
    left.axhline(chance, color="#c1440e", ls="--", label=f"chance {chance:.2f}")
    left.set_xlabel("layer")
    left.set_ylabel("overlap with the previous token")
    left.set_title("expert reuse, consecutive tokens")
    left.legend()

    right.plot(np.arange(1, curve.size + 1), curve, marker="o", color="#3c6e91")
    right.axhline(chance, color="#c1440e", ls="--", label=f"chance {chance:.2f}")
    right.set_xlabel("token distance")
    right.set_ylabel("mean overlap")
    right.set_title("reuse against distance")
    right.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_skew(trace: RouterTrace, out: Path) -> Path:
    skew = analysis.access_skew(trace)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    for layer in range(trace.n_layers):
        counts = np.sort(skew.counts[layer])[::-1]
        total = counts.sum()
        if total:
            left.plot(np.cumsum(counts) / total, color="#3c6e91", alpha=0.35, lw=1)
    left.plot([0, trace.n_experts - 1], [0, 1], color="#c1440e", ls="--", label="uniform router")
    left.set_xlabel("experts, busiest first")
    left.set_ylabel("cumulative share of routing")
    left.set_title("load imbalance, one line per layer")
    left.legend()

    right.bar(np.arange(trace.n_layers), skew.gini, color="#3c6e91")
    right.set_xlabel("layer")
    right.set_ylabel("gini coefficient")
    right.set_title("skew by layer, 0 uniform 1 concentrated")
    right.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_cache_curve(results: dict[str, list[cache_sim.SimResult]], total: int, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {"lru": "#c1440e", "lfu": "#e0a458", "belady": "#3c6e91"}
    for name, series in results.items():
        xs = [r.capacity / total for r in series]
        ys = [r.hit_rate for r in series]
        ax.plot(xs, ys, marker="o", label=name, color=colours.get(name))

    ax.axhline(0.7, color="#555", ls=":", label="g2 target, 0.70")
    ax.set_xscale("log")
    ax.set_xlabel("cache size, fraction of all expert-layer pairs")
    ax.set_ylabel("hit rate")
    ax.set_title("hit rate against ram budget, and where the knee is")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_coactivation(trace: RouterTrace, out: Path, layer: int | None = None) -> Path:
    layer = trace.n_layers // 2 if layer is None else layer
    result = analysis.coactivation(trace, layer)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(result.matrix, cmap="magma", interpolation="nearest")
    ax.set_title(
        f"co-activation, layer {layer}\n"
        f"lift {result.lift:.2f}x over independent, "
        f"top pairs hold {result.concentration:.1%}"
    )
    ax.set_xlabel("expert")
    ax.set_ylabel("expert")
    fig.colorbar(im, ax=ax, label="joint probability")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_prediction(results: list[predict.LookaheadResult], out: Path) -> Path:
    ks = [r.k for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [r.persistence for r in results], marker="s", color="#c1440e",
            label="persistence prior, free")
    ax.plot(ks, [r.linear for r in results], marker="o", color="#e0a458", label="linear probe")
    ax.plot(ks, [r.mlp for r in results], marker="^", color="#3c6e91", label="mlp probe")
    ax.axhline(0.9, color="#555", ls=":", label="m5 target, 0.90")
    ax.set_xlabel("layers of lookahead, k")
    ax.set_ylabel(f"recall at a budget of {results[0].budget} experts" if results else "recall")
    ax.set_title("can layer L+k routing be predicted from layer L?")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ report


def run(trace: RouterTrace, out_dir: str | Path, target_fraction: float = 0.2) -> Path:
    """run every analysis, write the plots and the report, return the report path.

    args:
        target_fraction: the ram budget the verdict is judged at, as a fraction
            of all expert-layer pairs. 0.2 is roughly a 16gb machine holding a
            120b class model's experts at four bits.
    """
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"

    reuse = analysis.reuse_across_tokens(trace)
    skew = analysis.access_skew(trace)
    coact = analysis.coactivation_summary(trace)

    total_pairs = trace.n_layers * trace.n_experts
    sim = cache_sim.sweep(trace)
    target_capacity = max(1, int(total_pairs * target_fraction))
    at_target = {
        name: min(series, key=lambda r: abs(r.capacity - target_capacity)).hit_rate
        for name, series in sim.items()
    }

    prediction: list[predict.LookaheadResult] = []
    if trace.has_hidden:
        prediction = predict.lookahead_recall(trace)

    plot_reuse(trace, figures / "reuse.png")
    plot_skew(trace, figures / "skew.png")
    plot_cache_curve(sim, total_pairs, figures / "cache_curve.png")
    plot_coactivation(trace, figures / "coactivation.png")
    if prediction:
        plot_prediction(prediction, figures / "prediction.png")

    chance = trace.top_k / trace.n_experts
    checks = [
        Check(
            "expert reuse across consecutive tokens",
            _verdict(reuse.overall, max(2 * chance, 0.3)),
            reuse.overall,
            max(2 * chance, 0.3),
            "if consecutive tokens route to unrelated experts, nothing can be cached",
        ),
        Check(
            "router load imbalance",
            _verdict(float(skew.gini.mean()), 0.3),
            float(skew.gini.mean()),
            0.3,
            "a uniform router pins the hit rate to the ram ratio and no policy helps",
        ),
        Check(
            "co-activation lift over independent routing",
            _verdict(coact["lift"], 1.3),
            coact["lift"],
            1.3,
            "without it, laying the file out by co-activation buys nothing",
        ),
        Check(
            f"achievable hit rate at {target_fraction:.0%} of experts resident",
            _verdict(at_target.get("belady", 0.0), 0.7),
            at_target.get("belady", 0.0),
            0.7,
            "g2 asks for 0.70, and belady is the ceiling any policy is measured against",
        ),
    ]

    k4 = next((r for r in prediction if r.k == 4), None)
    if k4 is not None:
        checks.append(
            Check(
                "layer L+4 routing predicted from layer L",
                _verdict(k4.best_probe(), 0.9),
                k4.best_probe(),
                0.9,
                "o2 is the project's research contribution and this is whether it exists",
            )
        )
        checks.append(
            Check(
                "probe beats the free persistence prior at k=4",
                PASS if k4.beats_baseline() else FAIL,
                k4.best_probe() - k4.persistence,
                0.0,
                "a probe that cannot beat a prior costing nothing should not be shipped",
            )
        )

    report = _write_markdown(
        trace, checks, reuse, skew, coact, sim, at_target, prediction, out_dir
    )

    summary = {
        "provenance": trace.provenance.to_dict(),
        "is_real_measurement": trace.provenance.is_real,
        "reuse_overall": reuse.overall,
        "gini_mean": float(skew.gini.mean()),
        "coactivation": coact,
        "hit_rate_at_target": at_target,
        "checks": [
            {"name": c.name, "verdict": c.verdict, "value": c.value, "threshold": c.threshold}
            for c in checks
        ],
        "prediction": [
            {
                "k": r.k,
                "budget": r.budget,
                "persistence": r.persistence,
                "linear": r.linear,
                "mlp": r.mlp,
            }
            for r in prediction
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report


def _write_markdown(
    trace: RouterTrace,
    checks: list[Check],
    reuse: analysis.ReuseResult,
    skew: analysis.SkewResult,
    coact: dict[str, float],
    sim: dict[str, list[cache_sim.SimResult]],
    at_target: dict[str, float],
    prediction: list[predict.LookaheadResult],
    out_dir: Path,
) -> Path:
    failed = [c for c in checks if c.verdict == FAIL]
    marginal = [c for c in checks if c.verdict == MARGINAL]

    if failed:
        headline = (
            f"**stop and reconsider.** {len(failed)} of {len(checks)} assumptions "
            "did not hold. the prd says to publish this as a short negative result "
            "and not to build the engine on it."
        )
    elif marginal:
        headline = (
            f"**proceed with a narrowed claim.** {len(marginal)} of {len(checks)} "
            "assumptions came out marginal. the risk table has a fallback for each."
        )
    else:
        headline = "**proceed.** every assumption the design rests on held."

    lines = [
        "# m0: does the structure strata depends on actually exist",
        "",
    ]

    if not trace.provenance.is_real:
        lines += [
            "> ## this is not a measurement",
            ">",
            f"> {trace.provenance.banner()}",
            ">",
            "> the numbers below describe a trace that was generated with the "
            "structure the design assumes, so of course it exhibits that structure. "
            "this run exercises the harness end to end. it says nothing whatsoever "
            "about any real model, and none of it belongs in a writeup.",
            "",
        ]
    else:
        lines += [f"*{trace.provenance.banner()}*", ""]

    lines += [
        "```",
        trace.describe(),
        "```",
        "",
        "## verdict",
        "",
        headline,
        "",
        "| check | verdict | measured | threshold | why it matters |",
        "|---|---|---|---|---|",
        *[c.line() for c in checks],
        "",
        "## 1. reuse across tokens",
        "",
        f"{reuse}. chance, if the router picked at random, would be "
        f"{trace.top_k / trace.n_experts:.3f}.",
        "",
        "![reuse](figures/reuse.png)",
        "",
        "the persistence prior is this number. it costs nothing, needs no model, "
        "and is the baseline the speculative router head has to beat before it is "
        "worth its complexity.",
        "",
        "## 2. access skew",
        "",
        f"{skew}",
        "",
        "![skew](figures/skew.png)",
        "",
        "## 3. hit rate against ram budget",
        "",
        "| policy | hit rate at the target budget | smallest cache reaching 0.70 |",
        "|---|---|---|",
    ]

    total_pairs = trace.n_layers * trace.n_experts
    for name, series in sim.items():
        k = cache_sim.knee(series, 0.70)
        knee_text = f"{k} pairs ({k / total_pairs:.1%})" if k else "never reaches it"
        lines.append(f"| {name} | {at_target.get(name, 0.0):.3f} | {knee_text} |")

    lines += [
        "",
        "![cache curve](figures/cache_curve.png)",
        "",
        "this is the figure that answers the question every actual user has, which "
        "is how much ram they need for their model.",
        "",
        "## 4. co-activation structure",
        "",
        f"joint routing runs at **{coact['lift']:.2f}x** what independent routing "
        f"would produce, and the heaviest pairs account for "
        f"**{coact['concentration']:.1%}** of all joint mass.",
        "",
        "![co-activation](figures/coactivation.png)",
        "",
        "lift near 1.0 would mean experts fire independently, in which case laying "
        "the file out by co-activation buys nothing and the storage design loses "
        "one of its two arguments.",
        "",
        "## 5. multi-layer-ahead predictability",
        "",
    ]

    if prediction:
        lines += [
            "| k | prefetch budget | persistence prior | linear probe | mlp probe |",
            "|---|---|---|---|---|",
            *[
                f"| {r.k} | {r.budget} | {r.persistence:.3f} | {r.linear:.3f} | {r.mlp:.3f} |"
                for r in prediction
            ],
            "",
            "![prediction](figures/prediction.png)",
            "",
            "recall, not accuracy, because a false positive costs bandwidth and a "
            "false negative costs a full stall. prefetching a superset is fine.",
        ]
    else:
        lines += [
            "not measured: this trace carries no hidden states. capture with "
            "`--hidden` to answer the question this project's research contribution "
            "depends on.",
        ]

    lines += [
        "",
        "---",
        "",
        "generated by `strata_m0.report`. the raw numbers are in `summary.json`.",
        "",
    ]

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
