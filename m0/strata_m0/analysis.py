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


# ------------------------------------------------------------ domain structure


@dataclass
class DomainResult:
    """how much the expert set depends on what is being talked about."""

    within: float
    """mean routing similarity between two token windows from the same subject."""

    across: float
    """mean similarity between two windows from different subjects."""

    @property
    def separation(self) -> float:
        """how much higher within-domain overlap is than across-domain.

        this is the number the cache policy rests on. the prd says a coding
        conversation hits a stable subset and a different subset dominates for a
        different domain, and that pure recency throws that structure away at
        every topic switch. if this is near zero, the frequency term in the
        eviction score is doing nothing that recency could not do, and lru is
        the right policy after all.
        """
        return self.within - self.across

    def __str__(self) -> str:
        return (
            f"domain correlation: {self.within:.3f} similarity within a subject, "
            f"{self.across:.3f} across, separation {self.separation:+.3f}"
        )


def _window_profiles(
    trace: RouterTrace,
    boundaries: list[int],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """routing count profile for every window that fits inside one segment.

    a window never straddles a subject boundary, so a window is always
    attributable to exactly one subject. returns the token index each window
    starts at, the subject it belongs to, and the l2 normalised counts over
    expert-layer pairs.
    """
    n_pairs = trace.n_layers * trace.n_experts
    edges = sorted(set(list(boundaries) + [trace.n_tokens]))
    layer_ids = np.arange(trace.n_layers)[None, :, None]

    starts: list[int] = []
    labels: list[int] = []
    profiles: list[np.ndarray] = []
    for domain, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        for w0 in range(start, end - window + 1, window):
            block = trace.routing[w0 : w0 + window]
            flat = np.broadcast_to(layer_ids, block.shape) * trace.n_experts + block
            counts = np.bincount(flat.reshape(-1), minlength=n_pairs).astype(np.float64)
            norm = np.linalg.norm(counts)
            if norm > 0:
                starts.append(w0)
                labels.append(domain)
                profiles.append(counts / norm)

    if not profiles:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros((0, n_pairs))
    return np.array(starts), np.array(labels), np.stack(profiles)


def _separation(labels: np.ndarray, gram: np.ndarray) -> tuple[float, float]:
    """mean similarity within a label against across labels.

    takes a precomputed gram matrix because the null resamples the labels
    hundreds of times against the same profiles, and recomputing the pairwise
    similarities every time is the whole cost.
    """
    n = len(labels)
    if n < 2:
        return 0.0, 0.0
    rows, cols = np.triu_indices(n, 1)
    same = labels[rows] == labels[cols]
    if not same.any() or same.all():
        return 0.0, 0.0
    values = gram[rows, cols]
    return float(values[same].mean()), float(values[~same].mean())


def domain_correlation(
    trace: RouterTrace,
    boundaries: list[int],
    window: int = 64,
) -> DomainResult:
    """compare routing similarity within a subject against across subjects.

    args:
        boundaries: token index where each subject starts, ascending, beginning
            at 0. a corpus of four topics gives four entries.
        window: tokens per window. large enough to have a characteristic
            profile, small enough that several fit inside one subject.

    the comparison is between windows of equal size in both cases, so the only
    thing that differs is whether the two came from the same subject.

    # two mistakes this deliberately avoids

    **it compares expert-layer pairs, not experts.** the first version took
    ``np.unique`` over a window of the whole routing tensor, which flattens
    every layer together. expert 5 of layer 3 and expert 5 of layer 30 are
    unrelated tensors, so that is the mistake the ``ExpertKey`` type exists to
    prevent, made in the analysis instead of the engine.

    **it compares distributions, not sets.** set overlap saturates. a 64 token
    window at 24 layers and top-8 draws twelve thousand times from 768 pairs, so
    every pair appears at least once and the jaccard index between any two
    windows is exactly 1.0. that is what the first version reported for both
    within and across, and a metric that returns 1.000 for everything is not
    measuring anything. cosine similarity over access *counts* keeps the
    information about which pairs are used heavily.

    the separation this returns is not interpretable on its own, because cosine
    over counts is dominated by the base rate every window shares. read it next
    to ``domain_null``.
    """
    if len(boundaries) < 2:
        return DomainResult(0.0, 0.0)
    _, labels, profiles = _window_profiles(trace, boundaries, window)
    if len(labels) < 2:
        return DomainResult(0.0, 0.0)
    within, across = _separation(labels, profiles @ profiles.T)
    return DomainResult(within=within, across=across)


@dataclass
class DomainNull:
    """the separation a corpus with no subject structure would have produced."""

    observed: float
    """separation measured against the real subject boundaries."""

    null_mean: float
    """mean separation under the circular shift null."""

    null_p95: float
    """95th percentile of the null. the bar the observed value has to clear."""

    exceeded: int
    """how many shifts produced a separation at least as large as observed."""

    n_shifts: int

    aligned: int = 0
    """shifts discarded because they reproduced the real subject boundaries.

    when the subjects are near enough to equal length, a shift by one subject
    length maps every block onto another block and the induced labelling is the
    true one with the labels renamed. that draw scores the observed separation
    against a copy of itself, so keeping it puts the alternative hypothesis
    inside the null and costs the test most of its power. the synthetic case
    with four exactly equal blocks made this obvious: a sixteenth of all shifts
    realigned perfectly and the null p95 came out exactly equal to the observed
    value, so no amount of planted structure could ever have been detected.
    """

    @property
    def margin(self) -> float:
        """how far the observed separation clears the null's 95th percentile."""
        return self.observed - self.null_p95

    @property
    def p_value(self) -> float:
        return (self.exceeded + 1) / (self.n_shifts + 1)

    def __str__(self) -> str:
        return (
            f"domain null: observed {self.observed:+.3f}, null mean "
            f"{self.null_mean:+.3f}, null p95 {self.null_p95:+.3f}, "
            f"p={self.p_value:.3f} over {self.n_shifts} shifts "
            f"({self.aligned} discarded as realignments)"
        )


def _partition(labels: np.ndarray) -> tuple[int, ...]:
    """labels rewritten so the naming of the subjects cannot matter.

    two labellings that group the same windows together are the same partition
    even if the subjects are numbered differently, and the separation statistic
    only depends on the grouping.
    """
    seen: dict[int, int] = {}
    out: list[int] = []
    for value in labels.tolist():
        if value not in seen:
            seen[value] = len(seen)
        out.append(seen[value])
    return tuple(out)


def domain_null(
    trace: RouterTrace,
    boundaries: list[int],
    window: int = 64,
    n_shifts: int = 200,
    seed: int = 0,
) -> DomainNull:
    """test the subject separation against a null that keeps everything but the subjects.

    # why a null is needed at all

    the separation from ``domain_correlation`` is a difference between two
    cosines that both sit near 1.0, because every window shares the router's
    base rate. the raw number is therefore not comparable against a threshold
    anyone could pick in advance, and the threshold it was being compared
    against had been chosen for a set-overlap metric that turned out to be
    broken. a number that only looks meaningful next to a threshold invented
    after the fact is not a measurement.

    # why a circular shift and not a label shuffle

    windows inside one subject are also adjacent in time, so a plain shuffle of
    the labels would confound subject matter with temporal drift and report
    structure that is really only locality. shifting the window positions
    circularly against fixed boundaries keeps the block sizes, the contiguity
    and the time distances intact, and destroys only the alignment between the
    blocks and the actual subjects. what survives that is subject matter.
    """
    if len(boundaries) < 2:
        return DomainNull(0.0, 0.0, 0.0, 0, 0)

    starts, labels, profiles = _window_profiles(trace, boundaries, window)
    if len(labels) < 2:
        return DomainNull(0.0, 0.0, 0.0, 0, 0)

    gram = profiles @ profiles.T
    within, across = _separation(labels, gram)
    observed = within - across

    edges = np.array(sorted(set(list(boundaries) + [trace.n_tokens])))
    truth = _partition(labels)
    rng = np.random.default_rng(seed)

    nulls: list[float] = []
    aligned = 0
    attempts = 0
    budget = 50 * n_shifts
    while len(nulls) < n_shifts and attempts < budget:
        attempts += 1
        shift = int(rng.integers(1, trace.n_tokens))
        moved = (starts + shift) % trace.n_tokens
        shifted = np.searchsorted(edges, moved, side="right") - 1
        if _partition(shifted) == truth:
            aligned += 1
            continue
        w, a = _separation(shifted, gram)
        nulls.append(w - a)

    if not nulls:
        return DomainNull(observed, 0.0, 0.0, 0, 0, aligned)

    draws = np.array(nulls)
    return DomainNull(
        observed=observed,
        null_mean=float(draws.mean()),
        null_p95=float(np.percentile(draws, 95)),
        exceeded=int((draws >= observed).sum()),
        n_shifts=len(draws),
        aligned=aligned,
    )
