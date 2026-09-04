"""hit rate against cache size, which is the second money graph.

the question a buyer actually has is not "how good is your policy" but "how much
ram do i need for my model". this module answers it: replay the trace at a range
of cache sizes and find the knee.

three policies, because one number in isolation says nothing. lru is the floor
every system in this space reports against, belady's optimum is the ceiling that
separates a bad policy from a cache that is simply too small, and lfu shows how
much of the achievable gain comes from frequency alone.

experts are treated as uniform in size here so that capacity can be quoted in
experts, which is the unit the reader can reason about, and so that belady is
exactly optimal rather than an approximation. the shipping policy is implemented
in rust in ``strata-cache`` and is replayed against the same traces there; this
module is deliberately only the baselines.
"""

from __future__ import annotations

import heapq
from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np

from .trace import RouterTrace


@dataclass
class SimResult:
    """one policy at one capacity."""

    policy: str
    capacity: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def lru(keys: np.ndarray, capacity: int) -> SimResult:
    """least recently used, admitting on first touch."""
    if capacity <= 0:
        return SimResult("lru", capacity, 0, int(keys.size))
    resident: OrderedDict[int, None] = OrderedDict()
    hits = misses = 0
    for key in keys.tolist():
        if key in resident:
            resident.move_to_end(key)
            hits += 1
            continue
        misses += 1
        resident[key] = None
        if len(resident) > capacity:
            resident.popitem(last=False)
    return SimResult("lru", capacity, hits, misses)


def lfu(keys: np.ndarray, capacity: int) -> SimResult:
    """least frequently used over the whole run.

    included because it is the other obvious thing to try, and because the gap
    between it and lru says which signal a workload actually carries. it has no
    decay at all, so it ossifies on a topic switch, which is a failure worth
    seeing rather than a reason to leave it out.
    """
    if capacity <= 0:
        return SimResult("lfu", capacity, 0, int(keys.size))
    freq: Counter[int] = Counter()
    resident: set[int] = set()
    hits = misses = 0
    for key in keys.tolist():
        freq[key] += 1
        if key in resident:
            hits += 1
            continue
        misses += 1
        if len(resident) >= capacity:
            victim = min(resident, key=lambda k: (freq[k], k))
            if freq[victim] > freq[key]:
                continue
            resident.discard(victim)
        resident.add(key)
    return SimResult("lfu", capacity, hits, misses)


def belady(keys: np.ndarray, capacity: int) -> SimResult:
    """evict whatever is used furthest in the future.

    not implementable online, which is the point: it says how much of the miss
    rate is the policy's fault and how much is the cache simply being too small.
    exactly optimal here because every expert is treated as the same size.
    """
    if capacity <= 0:
        return SimResult("belady", capacity, 0, int(keys.size))

    seq = keys.tolist()
    positions: dict[int, list[int]] = {}
    for i, key in enumerate(seq):
        positions.setdefault(key, []).append(i)
    cursor: dict[int, int] = dict.fromkeys(positions, 0)

    def next_use(key: int) -> int:
        uses = positions[key]
        c = cursor[key]
        return uses[c] if c < len(uses) else len(seq) + 1

    resident: set[int] = set()
    # max heap on next use, with lazy invalidation of stale entries
    horizon: list[tuple[int, int]] = []
    hits = misses = 0

    for i, key in enumerate(seq):
        cursor[key] += 1
        if key in resident:
            hits += 1
            heapq.heappush(horizon, (-next_use(key), key))
            continue
        misses += 1
        while len(resident) >= capacity:
            neg, victim = heapq.heappop(horizon)
            if victim not in resident or -neg != next_use(victim):
                continue  # stale
            resident.discard(victim)
        resident.add(key)
        heapq.heappush(horizon, (-next_use(key), key))
    return SimResult("belady", capacity, hits, misses)


POLICIES = {"lru": lru, "lfu": lfu, "belady": belady}


def sweep(
    trace: RouterTrace,
    capacities: list[int] | None = None,
    policies: list[str] | None = None,
) -> dict[str, list[SimResult]]:
    """hit rate at a range of cache sizes, for each policy.

    capacities are in expert-layer pairs. the total number of distinct pairs in
    the model is ``n_layers * n_experts``, so the sweep spans from a cache that
    holds almost nothing to one that holds everything, which is what makes the
    knee visible.
    """
    keys = trace.flat_keys()
    total = trace.n_layers * trace.n_experts
    if capacities is None:
        capacities = sorted(
            {max(1, int(total * f)) for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)}
        )
    if policies is None:
        policies = ["lru", "lfu", "belady"]

    return {
        name: [POLICIES[name](keys, cap) for cap in capacities] for name in policies
    }


def knee(results: list[SimResult], target: float = 0.7) -> int | None:
    """smallest capacity that reaches `target` hit rate, if any does.

    g2 in the prd asks for 70 percent. this is the function that says how much
    ram that costs, which is the number a reader actually wants.
    """
    for r in sorted(results, key=lambda r: r.capacity):
        if r.hit_rate >= target:
            return r.capacity
    return None
