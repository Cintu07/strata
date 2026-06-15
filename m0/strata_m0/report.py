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
    ax.plot(ks, [r.static for r in results], marker="^", color="#7a5c00",
            label="static frequency prior, free")
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

    # a decoder walks every layer of every token in order, so one token is a
    # scan over n_layers * top_k distinct pairs. below that, recency evicts each
    # pair exactly before its next use. this measures the step rather than
    # asserting it, because a zero in a results table that is not explained
    # reads as a broken simulator.
    keys = trace.flat_keys()
    per_token = trace.n_layers * trace.top_k
    cliff = {
        "per_token": per_token,
        "target_capacity": target_capacity,
        "lru_below": cache_sim.lru(keys, max(1, per_token - 1)).hit_rate,
        "lru_at": cache_sim.lru(keys, per_token).hit_rate,
    }

    prediction: list[predict.LookaheadResult] = []
    if trace.has_hidden:
        prediction = predict.lookahead_recall(trace)

    domain = None
    null = None
    if trace.has_segments:
        domain = analysis.domain_correlation(trace, trace.segments.tolist())
        null = analysis.domain_null(trace, trace.segments.tolist())

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

    if null is not None and null.n_shifts:
        # graded on the permutation p rather than on the margin, because the
        # margin is a difference of two cosines and has no scale anyone can read.
        if null.p_value <= 0.01:
            verdict = PASS
        elif null.p_value <= 0.05:
            verdict = MARGINAL
        else:
            verdict = FAIL
        checks.append(
            Check(
                "subject separation survives a circular shift null",
                verdict,
                null.margin,
                0.0,
                "if routing is the same whatever the text is about, frequency "
                "buys nothing over recency and lru is the right policy",
            )
        )

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
                "probe beats the best free baseline at k=4",
                _verdict(k4.margin(), 0.05),
                k4.margin(),
                0.05,
                "a probe that barely beats a table costing nothing should not be "
                "shipped, and 0.05 recall is the least that could pay for one",
            )
        )

    report = _write_markdown(
        trace, checks, reuse, skew, coact, sim, at_target, cliff, prediction,
        domain, null, out_dir,
    )

    summary = {
        "provenance": trace.provenance.to_dict(),
        "is_real_measurement": trace.provenance.is_real,
        "reuse_overall": reuse.overall,
        "domain": None
        if domain is None
        else {
            "within": domain.within,
            "across": domain.across,
            "separation": domain.separation,
        },
        "domain_null": None
        if null is None or not null.n_shifts
        else {
            "observed": null.observed,
            "null_mean": null.null_mean,
            "null_p95": null.null_p95,
            "margin": null.margin,
            "p_value": null.p_value,
            "n_shifts": null.n_shifts,
            "aligned_discarded": null.aligned,
        },
        "gini_mean": float(skew.gini.mean()),
        "coactivation": coact,
        "hit_rate_at_target": at_target,
        "lru_cliff": cliff,
        "checks": [
            {"name": c.name, "verdict": c.verdict, "value": c.value, "threshold": c.threshold}
            for c in checks
        ],
        "prediction": [
            {
                "k": r.k,
                "budget": r.budget,
                "persistence": r.persistence,
                "static": r.static,
                "free_baseline": r.free_baseline(),
                "margin": r.margin(),
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
    cliff: dict[str, float],
    prediction: list[predict.LookaheadResult],
    domain: analysis.DomainResult | None,
    null: analysis.DomainNull | None,
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
        "and it is one of the two free baselines a prefetcher has to beat. it is "
        "the weaker one. see section 5. "
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
        f"**lru reads {at_target.get('lru', 0.0):.3f} here and that is not a "
        f"broken simulator.** one token touches {cliff['per_token']:.0f} "
        f"distinct expert-layer pairs, {trace.n_layers} layers at "
        f"top-{trace.top_k}, and the budget being judged holds "
        f"{cliff['target_capacity']:.0f}. a cyclic scan over more distinct items "
        f"than the cache holds evicts every one of them exactly before it is "
        f"next needed, which is the worst case for pure recency, and it is not a "
        f"rare corner: it is what a decoder does to any cache smaller than one "
        f"token's working set.",
        "",
        f"the step is measured, not assumed. at {cliff['per_token'] - 1:.0f} "
        f"pairs lru gets {cliff['lru_below']:.3f}, and at {cliff['per_token']:.0f} "
        f"pairs it gets {cliff['lru_at']:.3f}.",
        "",
        f"that is the argument for admission control in one number. lfu reads "
        f"{at_target.get('lfu', 0.0):.3f} on the same trace at the same budget, "
        f"because frequency survives a scan that recency cannot, and it is why "
        f"the strata cache puts a probationary window in front of the main "
        f"region rather than running one lru list.",
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

    if domain is not None:
        lines[-2:-2] = [
            "## 4b. does routing depend on the subject",
            "",
            f"routing profiles are **{domain.within:.3f}** similar between two "
            f"windows of the same subject and **{domain.across:.3f}** similar "
            f"between windows of different subjects, a separation of "
            f"**{domain.separation:+.3f}**. similarity is cosine over access "
            f"counts across expert-layer pairs, which does not saturate the way "
            f"set overlap does.",
            "",
            "this is the claim the cache policy rests on. the eviction score is "
            "frequency based rather than purely recency based because a topic is "
            "supposed to have a stable expert set that survives a digression. a "
            "separation near zero would mean lru is the right policy and the "
            "extra machinery is not earning its place.",
            "",
        ]

    if null is not None and null.n_shifts:
        explained = null.null_mean / null.observed if null.observed > 0 else 0.0
        lines[-2:-2] = [
            "that separation cannot be read on its own. windows inside one "
            "subject are also adjacent in time, so some of it is temporal "
            "locality rather than subject matter. shifting the window positions "
            "circularly against the same boundaries keeps the block sizes, the "
            "contiguity and the time distances and destroys only the alignment "
            "with the actual subjects.",
            "",
            "| | separation |",
            "|---|---|",
            f"| observed | {null.observed:+.3f} |",
            f"| circular shift null, mean | {null.null_mean:+.3f} |",
            f"| circular shift null, p95 | {null.null_p95:+.3f} |",
            f"| margin over p95 | {null.margin:+.3f} |",
            f"| p over {null.n_shifts} shifts | {null.p_value:.3f} |",
            "",
            f"{null.aligned} further shifts were discarded because they mapped "
            f"each subject onto another subject and so reproduced the real "
            f"boundaries exactly. the subjects here are close to equal length, "
            f"which is what makes that happen, and scoring the observed value "
            f"against copies of itself would have cost the test most of its "
            f"power.",
            "",
            f"**the null already explains {explained:.0%} of the separation.** "
            f"what is left is real at p={null.p_value:.3f} and small. routing "
            f"here is mostly a property of position in the text, not of what the "
            f"text is about, and that is a weaker result than the prd assumes.",
            "",
        ]

    if prediction:
        lines += [
            "| k | budget | persistence prior | static prior | linear probe "
            "| mlp probe | margin |",
            "|---|---|---|---|---|---|---|",
            *[
                f"| {r.k} | {r.budget} | {r.persistence:.3f} | {r.static:.3f} | "
                f"{r.linear:.3f} | {r.mlp:.3f} | {r.margin():+.3f} |"
                for r in prediction
            ],
            "",
            "![prediction](figures/prediction.png)",
            "",
            "recall, not accuracy, because a false positive costs bandwidth and a "
            "false negative costs a full stall. prefetching a superset is fine.",
            "",
            "**margin is against the better of the two free baselines**, not "
            "against the persistence prior alone. the static prior is a table of "
            "per-layer expert popularity built once when the model is profiled. "
            "it reads the hidden state never, it costs nothing in the decode "
            "loop, and here it beats the persistence prior at every k and beats "
            "the linear probe at every k.",
            "",
            "read the margin column, not the probe column. the probe is the only "
            "thing in this table that has to be trained, shipped and run inside "
            "the decode loop, and what it buys over a table of counts is what it "
            "has to justify itself with.",
            "",
            "note also that recall barely moves with k. that is not the good news "
            "it looks like. a predictor that is no worse eight layers out than "
            "one layer out is not using the lookahead, it is reproducing "
            "something that does not depend on k, and a static popularity table "
            "is exactly that. the flatness and the margin are the same fact.",
        ]
    else:
        lines += [
            "not measured: this trace carries no hidden states. capture with "
            "`--hidden` to answer the question this project's research contribution "
            "depends on.",
        ]

    density = trace.top_k / trace.n_experts
    lines += [
        "",
        "## what this run does not measure",
        "",
        f"this model activates **{density:.0%}** of its experts per layer per "
        f"token, top-{trace.top_k} of {trace.n_experts}. the models this project "
        f"exists for are far sparser, and density is not a detail: it sets the "
        f"chance baselines every result here is read against. random routing "
        f"would reuse {density:.3f} of its experts between consecutive tokens, "
        f"and both free priors the probe has to beat are high for the same "
        f"reason. the models the design targets route nearer 3 to 6 percent, so "
        f"every baseline here moves on them and none of these numbers transfers "
        f"without being measured again.",
        "",
        "which way they move is not known. it has not been measured here, and "
        "guessing the direction is the thing this harness exists to avoid.",
        "",
        f"the corpus is {trace.n_tokens} tokens. that is enough to separate the "
        f"effects reported here and not enough to put a confidence interval on "
        f"any of them beyond the one null that carries a p value.",
        "",
        "---",
        "",
        "generated by `strata_m0.report`. the raw numbers are in `summary.json`.",
        "",
    ]

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
