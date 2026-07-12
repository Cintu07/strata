"""how hard did we try before calling layer L+k unpredictable.

a negative result is only worth publishing if the thing it says does not exist
was looked for properly. m0 reports that a probe reading layer L cannot predict
layer L+4's routing well enough to prefetch on, and the two obvious rebuttals
are that the input was crushed by the capture's random projection and that the
probe was too small to learn anything.

this sweeps both. run it against two traces of the same tokens captured at
different projection widths, and against a range of probe capacities, and print
what recall each reaches. if recall is flat across an order of magnitude of
probe capacity and across a tenfold change in input width, the ceiling is in the
model and not in the estimator.

    python scripts/probe_capacity.py traces/granite.npz traces/granite-full.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_m0 import predict, trace as trace_mod

K = 4

# hidden units, epochs. 512 takes about six minutes on this machine and the
# larger settings grow with the product of the two, so this stops where the
# trend is already unambiguous rather than where it flattens. recall was still
# climbing at 512, which is reported rather than papered over.
CAPACITIES = [
    (32, 60),
    (128, 60),
    (512, 60),
]


def rows_for_k(trace: trace_mod.RouterTrace, k: int):
    """the same pooled design matrix lookahead_recall builds, for one k."""
    xs, ys, priors = [], [], []
    for layer in range(trace.n_layers - k):
        xs.append(trace.hidden[1:, layer, :])
        ys.append(predict._multi_hot(trace.routing[1:, layer + k, :], trace.n_experts))
        priors.append(
            predict._multi_hot(trace.routing[:-1, layer + k, :], trace.n_experts)
        )

    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    prior = np.concatenate(priors)

    n_per_layer = trace.n_tokens - 1
    train, test = predict._split(n_per_layer, 0.7)
    mask_train = np.zeros(x.shape[0], dtype=bool)
    mask_test = np.zeros(x.shape[0], dtype=bool)
    for layer in range(trace.n_layers - k):
        base = layer * n_per_layer
        mask_train[base + train.start : base + train.stop] = True
        mask_test[base + test.start : base + test.stop] = True

    return x, y, prior, mask_train, mask_test


def main(paths: list[str]) -> int:
    budget = None
    for path in paths:
        trace = trace_mod.RouterTrace.load(Path(path))
        if not trace.has_hidden:
            print(f"{path}: no hidden states, skipping")
            continue

        x, y, prior, mask_train, mask_test = rows_for_k(trace, K)
        budget = max(1, int(round(trace.top_k * 2.0)))
        x_tr, y_tr = x[mask_train], y[mask_train]
        x_te, y_te = x[mask_test], y[mask_test]

        persistence = predict.recall_at(prior[mask_test], y_te, budget)

        # the free baseline that actually matters: a table of per-layer expert
        # popularity, counted on the training rows only
        static_scores = np.zeros_like(y, dtype=np.float32)
        n_per_layer = trace.n_tokens - 1
        train, _ = predict._split(n_per_layer, 0.7)
        for layer in range(trace.n_layers - K):
            base = layer * n_per_layer
            counts = y[base + train.start : base + train.stop].sum(axis=0)
            static_scores[base : base + n_per_layer] = counts
        static = predict.recall_at(static_scores[mask_test], y_te, budget)
        free = max(persistence, static)

        linear = predict.recall_at(
            predict.LinearProbe().fit(x_tr, y_tr).predict(x_te), y_te, budget
        )

        print()
        print(f"== {path}")
        print(
            f"   {trace.n_tokens} tokens, hidden width {trace.hidden.shape[-1]}, "
            f"{x_tr.shape[0]} train rows, recall at k={K} into a budget of {budget}"
        )
        print(f"   {'probe':>18}  {'recall':>7}  {'over free':>10}  {'sec':>6}")
        print(f"   {'persistence prior':>18}  {persistence:7.4f}  {'':>10}  {'':>6}")
        print(f"   {'static freq prior':>18}  {static:7.4f}  {'':>10}  {'':>6}")
        print(f"   {'linear':>18}  {linear:7.4f}  {linear - free:+10.4f}  {'':>6}")

        for hidden, epochs in CAPACITIES:
            t0 = time.time()
            probe = predict.MlpProbe(hidden=hidden, epochs=epochs, seed=0)
            recall = predict.recall_at(
                probe.fit(x_tr, y_tr).predict(x_te), y_te, budget
            )
            label = f"mlp {hidden}x{epochs}"
            print(
                f"   {label:>18}  {recall:7.4f}  {recall - free:+10.4f}  "
                f"{time.time() - t0:6.1f}"
            )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["traces/granite.npz", "traces/granite-full.npz"]
    raise SystemExit(main(args))
