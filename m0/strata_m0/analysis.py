"""the three structural questions m0 asks of a trace.

each function here answers one question the whole design rests on, and each one
can come back with an answer that kills the project. that is the point of m0.

1. **is there reuse across tokens?** if consecutive tokens route to unrelated
   experts, nothing can be cached and no amount of policy work helps.
2. **is access skewed?** if every expert is equally likely, the cache can only
   ever hold its own fraction of the model and the hit rate is pinned to the
   ram ratio.
3. **is there co-activation structure?** if experts fire independently, laying
   the file out by co-activation buys nothing and the storage design loses one
   of its two arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import RouterTrace


# ------------------------------------------------------------------- reuse


@dataclass
class ReuseResult:
    """how much of one token's expert set the next token also wants."""

    per_layer: np.ndarray
    """mean overlap fraction for each layer."""

    overall: float
    """mean across layers, weighted equally."""

    def __str__(self) -> str:
        return (
            f"reuse across consecutive tokens: {self.overall:.3f} overall, "
            f"{self.per_layer.min():.3f} to {self.per_layer.max():.3f} across layers"
        )


def reuse_across_tokens(trace: RouterTrace, distance: int = 1) -> ReuseResult:
    """fraction of a token's experts that the token `distance` ahead also uses.

    this is the persistence prior stated as a number, and it is both the first
    go/no-go check and the baseline any predictor has to beat. it costs nothing
    to compute and nothing to deploy, so a speculative router head that does not
    clear it is not worth shipping.
    """
    if trace.n_tokens <= distance:
        return ReuseResult(np.zeros(trace.n_layers), 0.0)

    per_layer = np.zeros(trace.n_layers, dtype=np.float64)
    for layer in range(trace.n_layers):
        routed = trace.layer(layer)
        overlaps = np.empty(trace.n_tokens - distance, dtype=np.float64)
        for t in range(trace.n_tokens - distance):
            a = np.unique(routed[t])
            b = np.unique(routed[t + distance])
            overlaps[t] = np.intersect1d(a, b, assume_unique=True).size / a.size
        per_layer[layer] = overlaps.mean()
    return ReuseResult(per_layer, float(per_layer.mean()))


def reuse_curve(trace: RouterTrace, max_distance: int = 16) -> np.ndarray:
    """reuse as a function of how far apart two tokens are.

    the shape matters more than any single value. a curve that falls off a cliff
    after one token means only the immediately previous token is informative. a
    curve with a long flat tail means a stable working set, which is what makes
    a cache worth having at all.
    """
    return np.array(
        [reuse_across_tokens(trace, d).overall for d in range(1, max_distance + 1)]
    )


# -------------------------------------------------------------------- skew


@dataclass
class SkewResult:
    """how unevenly the router spreads its load."""

    counts: np.ndarray
    """``[n_layers, n_experts]`` routing counts."""

    gini: np.ndarray
    """per layer gini coefficient, 0 uniform and 1 maximally concentrated."""

    normalised_entropy: np.ndarray
    """per layer entropy over log(n_experts), 1 uniform and 0 degenerate."""

    top_decile_mass: np.ndarray
    """per layer share of all routing taken by the busiest tenth of experts."""

    def __str__(self) -> str:
        return (
            f"access skew: gini {self.gini.mean():.3f}, "
            f"normalised entropy {self.normalised_entropy.mean():.3f}, "
            f"busiest tenth takes {self.top_decile_mass.mean():.1%} of routing"
        )


def access_counts(trace: RouterTrace) -> np.ndarray:
    """``[n_layers, n_experts]`` counts of how often each expert was routed."""
    counts = np.zeros((trace.n_layers, trace.n_experts), dtype=np.int64)
    for layer in range(trace.n_layers):
        flat = trace.layer(layer).reshape(-1)
        counts[layer] = np.bincount(flat, minlength=trace.n_experts)
    return counts


def _gini(x: np.ndarray) -> float:
    """gini coefficient of a non negative vector."""
    total = x.sum()
    if total == 0:
        return 0.0
    sorted_x = np.sort(x.astype(np.float64))
    n = sorted_x.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * sorted_x).sum()) / (n * total) - (n + 1.0) / n)


def access_skew(trace: RouterTrace) -> SkewResult:
    """measure how concentrated routing is, per layer.

    three statistics rather than one because they fail differently. gini is
    sensitive to the tail, entropy to the head, and the decile share is the one
    a reader can actually picture.
    """
    counts = access_counts(trace)
    n_layers, n_experts = counts.shape

    gini = np.array([_gini(counts[layer]) for layer in range(n_layers)])

    entropy = np.zeros(n_layers)
    for layer in range(n_layers):
        p = counts[layer].astype(np.float64)
        total = p.sum()
        if total == 0:
            continue
        p = p[p > 0] / total
        entropy[layer] = float(-(p * np.log(p)).sum() / np.log(n_experts))

    decile = max(1, n_experts // 10)
    top_mass = np.zeros(n_layers)
    for layer in range(n_layers):
        total = counts[layer].sum()
        if total == 0:
            continue
        top = np.sort(counts[layer])[::-1][:decile].sum()
        top_mass[layer] = float(top / total)

    return SkewResult(counts, gini, entropy, top_mass)


# ------------------------------------------------------------ co-activation


@dataclass
class CoactivationResult:
    """which experts fire on the same token."""

    matrix: np.ndarray
    """``[n_experts, n_experts]`` joint probability, diagonal zeroed."""

    concentration: float
    """share of all joint mass held by the heaviest ``n_experts`` pairs.

    the comparison that matters: with `n_experts` pairs you could give every
    expert one neighbour, so this is roughly the fraction of co-activation a
    single linear layout could hope to capture.
    """

    lift: float
    """how much more concentrated joint routing is than the marginals alone
    would produce.

    1.0 means experts fire independently given how often each is used, and the
    co-activation layout buys nothing. above 1.0 means there are pairs that seek
    each other out, which is what the layout pass exploits.

    note what this deliberately is not. the obvious formulation, total observed
    joint mass over total expected joint mass, cannot work here: with a fixed
    top-k every token contributes exactly ``C(k, 2)`` pairs, so the total is a
    constant and carries no information about structure at all. what varies is
    how that fixed mass is distributed, so the expected distribution is rescaled
    to the same total and the comparison is about shape only.
    """


def coactivation(trace: RouterTrace, layer: int) -> CoactivationResult:
    """the co-activation graph for one layer."""
    routed = trace.layer(layer)
    n = trace.n_experts
    joint = np.zeros((n, n), dtype=np.float64)

    for t in range(trace.n_tokens):
        experts = np.unique(routed[t])
        joint[np.ix_(experts, experts)] += 1.0
    np.fill_diagonal(joint, 0.0)
    joint /= max(trace.n_tokens, 1)

    upper = joint[np.triu_indices(n, k=1)]
    total = upper.sum()
    if total == 0:
        return CoactivationResult(joint, 0.0, 1.0)

    top = np.sort(upper)[::-1][:n].sum()
    concentration = float(top / total)

    # what independent routing would have produced, given the same marginals,
    # rescaled to the same total so the comparison is about shape and not about
    # a total that top-k pins to a constant
    marginal = access_counts(trace)[layer].astype(np.float64)
    marginal /= max(trace.n_tokens, 1)
    expected = np.outer(marginal, marginal)[np.triu_indices(n, k=1)]
    expected_total = expected.sum()
    if expected_total <= 0:
        return CoactivationResult(joint, concentration, 1.0)
    expected = expected * (total / expected_total)

    # mass weighted mean of the per pair ratio. weighting by observed mass is
    # what keeps the answer about the pairs that actually happen, rather than
    # letting thousands of pairs that never fire dominate by being individually
    # close to zero on both sides.
    live = expected > 0
    ratio = np.zeros_like(upper)
    ratio[live] = upper[live] / expected[live]
    lift = float((upper[live] * ratio[live]).sum() / total)

    return CoactivationResult(joint, concentration, lift)


def coactivation_summary(trace: RouterTrace) -> dict[str, float]:
    """concentration and lift averaged over every layer."""
    results = [coactivation(trace, layer) for layer in range(trace.n_layers)]
    return {
        "concentration": float(np.mean([r.concentration for r in results])),
        "lift": float(np.mean([r.lift for r in results])),
    }
