"""can layer L+k routing be predicted from layer L's hidden state?

this is the central empirical question of the whole project, and the reason m0
comes before any engine code.

nvme read latency is 50 to 150 microseconds. a layer of compute on a laptop may
take less than that. so a prefetcher that looks one layer ahead has already lost:
by the time layer L finishes and layer L+1's router runs, the read cannot arrive
in time. hiding the read needs three to eight layers of lookahead, and the true
expert set for layer L+k is not known until layer L+k-1 has finished.

so something has to guess. three properties make guessing tractable:

- **only the union matters.** prefetching a superset is fine. bandwidth is
  wasted and nothing else is lost.
- **being wrong is never a correctness issue.** this is prefetch, not
  speculative execution. the true router still runs and still decides. there is
  no rollback and no change to the output.
- **it can be wrong asymmetrically.** a false positive costs bandwidth, a false
  negative costs a full stall, so recall is what to optimise and precision is
  close to free.

what is measured here, therefore, is **recall at a fixed prefetch budget**: if
the system is willing to fetch `budget` experts per layer, how much of what is
actually needed does it get?

the baseline every probe must beat is the persistence prior, which is free:
assume layer L+k routes the same way layer L did. if a trained probe cannot beat
that, the speculative router head is not worth building and the project should
fall back to the persistence prior plus co-activation prefetch, as the prd's
risk table says.

both probes are numpy, deliberately. this is a measurement, not a model to ship,
and the answer should not depend on a training framework's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import RouterTrace


def _multi_hot(indices: np.ndarray, n_experts: int) -> np.ndarray:
    """``[n_tokens, top_k]`` of indices to ``[n_tokens, n_experts]`` of 0/1."""
    out = np.zeros((indices.shape[0], n_experts), dtype=np.float32)
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    out[rows, indices.reshape(-1)] = 1.0
    return out


def recall_at(scores: np.ndarray, truth: np.ndarray, budget: int) -> float:
    """mean fraction of the true expert set captured by the top `budget` scores.

    ties are broken by index rather than at random, so a rerun gives the same
    number.
    """
    if scores.shape[0] == 0:
        return 0.0
    budget = min(budget, scores.shape[1])
    top = np.argpartition(-scores, budget - 1, axis=1)[:, :budget]
    captured = np.take_along_axis(truth, top, axis=1).sum(axis=1)
    needed = truth.sum(axis=1)
    needed = np.maximum(needed, 1.0)
    return float((captured / needed).mean())


# ------------------------------------------------------------------- probes


class LinearProbe:
    """ridge regression from hidden state to a multi-hot expert set.

    the first thing to try, and the one that settles whether the information is
    there at all. closed form, so there is no training schedule to get wrong and
    no chance of reporting an optimisation failure as a negative result.
    """

    name = "linear"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearProbe":
        x = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float32)])
        gram = x.T @ x
        gram[np.diag_indices_from(gram)] += self.alpha
        self.weights = np.linalg.solve(gram, x.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.weights is not None, "fit before predict"
        x = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float32)])
        return x @ self.weights


class MlpProbe:
    """one hidden layer, trained with adam on a multi-label logistic loss.

    this is the shape the prd proposes for the shipped speculative router head:
    a few hundred kilobytes, trained once per model, shipped alongside it. the
    point of measuring it here is to find out whether the extra capacity over
    the linear probe buys anything before committing to build one.
    """

    name = "mlp"

    def __init__(
        self,
        hidden: int = 128,
        epochs: int = 60,
        lr: float = 3e-3,
        batch: int = 256,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed
        self.params: dict[str, np.ndarray] = {}

    def _init(self, d_in: int, d_out: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.params = {
            "w1": (rng.normal(size=(d_in, self.hidden)) / np.sqrt(d_in)).astype(np.float32),
            "b1": np.zeros(self.hidden, dtype=np.float32),
            "w2": (rng.normal(size=(self.hidden, d_out)) / np.sqrt(self.hidden)).astype(
                np.float32
            ),
            "b2": np.zeros(d_out, dtype=np.float32),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    def _step(self, grads: dict[str, np.ndarray]) -> None:
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            m_hat = self._m[k] / (1 - b1**self._t)
            v_hat = self._v[k] / (1 - b2**self._t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MlpProbe":
        self._init(x.shape[1], y.shape[1])
        rng = np.random.default_rng(self.seed)
        n = x.shape[0]
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                idx = order[start : start + self.batch]
                xb, yb = x[idx], y[idx]

                h_pre = xb @ self.params["w1"] + self.params["b1"]
                h = np.tanh(h_pre)
                logits = h @ self.params["w2"] + self.params["b2"]

                # gradient of mean binary cross entropy with logits
                probs = 1.0 / (1.0 + np.exp(-logits))
                d_logits = (probs - yb) / xb.shape[0]
                d_h = d_logits @ self.params["w2"].T
                d_h_pre = d_h * (1.0 - h * h)

                self._step(
                    {
                        "w2": h.T @ d_logits,
                        "b2": d_logits.sum(axis=0),
                        "w1": xb.T @ d_h_pre,
                        "b1": d_h_pre.sum(axis=0),
                    }
                )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return h @ self.params["w2"] + self.params["b2"]


# ---------------------------------------------------------------- evaluation


@dataclass
class LookaheadResult:
    """recall at one lookahead distance."""

    k: int
    budget: int
    persistence: float
    static: float
    linear: float
    mlp: float

    def best_probe(self) -> float:
        return max(self.linear, self.mlp)

    def free_baseline(self) -> float:
        """the best recall obtainable without a probe at all.

        both of these cost nothing at inference time, so a shipped probe has to
        beat whichever is higher, not whichever is more flattering. the
        persistence prior was the only one m0 originally compared against, and
        on granite it is the weaker of the two by a wide margin, which made the
        probe look better than it is.
        """
        return max(self.persistence, self.static)

    def margin(self) -> float:
        return self.best_probe() - self.free_baseline()

    def beats_baseline(self) -> bool:
        return self.margin() > 0.0


def _split(n: int, train_frac: float) -> tuple[slice, slice]:
    """contiguous train and test split.

    contiguous rather than random on purpose. adjacent tokens are strongly
    correlated, so a random split puts near duplicates of the test set into
    training and reports a recall the deployed system would never see.
    """
    cut = int(n * train_frac)
    return slice(0, cut), slice(cut, n)


def lookahead_recall(
    trace: RouterTrace,
    k_values: tuple[int, ...] = (1, 2, 4, 8),
    budget_multiplier: float = 2.0,
    train_frac: float = 0.7,
    seed: int = 0,
) -> list[LookaheadResult]:
    """recall of layer L+k routing predicted from layer L's hidden state.

    args:
        budget_multiplier: how many experts may be prefetched, as a multiple of
            top_k. at 2.0 the system fetches twice what it needs and hopes the
            right half is in there.

    raises:
        ValueError: if the trace has no hidden states, since there is nothing to
            predict from.
    """
    if not trace.has_hidden:
        raise ValueError(
            "this trace has no hidden states, so multi-layer-ahead prediction "
            "cannot be measured. capture with --hidden."
        )

    budget = max(1, int(round(trace.top_k * budget_multiplier)))
    train, test = _split(trace.n_tokens - 1, train_frac)
    results: list[LookaheadResult] = []

    for k in k_values:
        if k >= trace.n_layers:
            continue

        # pool every (L, L+k) pair so one probe is learned for the whole model,
        # which is what would actually be shipped.
        #
        # tokens start at 1 because the baseline needs a previous token.
        xs, ys, persistence_scores = [], [], []
        for layer in range(trace.n_layers - k):
            xs.append(trace.hidden[1:, layer, :])
            ys.append(_multi_hot(trace.routing[1:, layer + k, :], trace.n_experts))
            # the free baseline: when the engine is at layer L of token t and
            # wants layer L+k, the thing it already knows for nothing is what
            # token t-1 routed to at layer L+k. that is the persistence prior,
            # and it is across tokens at a fixed layer rather than across layers.
            persistence_scores.append(
                _multi_hot(trace.routing[:-1, layer + k, :], trace.n_experts)
            )

        x = np.concatenate(xs).astype(np.float32)
        y = np.concatenate(ys)
        prior = np.concatenate(persistence_scores)

        # the split has to be applied per source layer, not to the pooled array,
        # or training tokens from one layer leak into the test set of another
        n_per_layer = trace.n_tokens - 1
        n_layers_used = trace.n_layers - k
        mask_train = np.zeros(x.shape[0], dtype=bool)
        mask_test = np.zeros(x.shape[0], dtype=bool)
        for layer in range(n_layers_used):
            base = layer * n_per_layer
            mask_train[base + train.start : base + train.stop] = True
            mask_test[base + test.start : base + test.stop] = True

        x_tr, y_tr = x[mask_train], y[mask_train]
        x_te, y_te = x[mask_test], y[mask_test]

        # the prior needs no training, so it is scored on the same test rows
        persistence = recall_at(prior[mask_test], y_te, budget)

        # the other free baseline: which experts are simply the most popular at
        # layer L+k. it ignores the hidden state completely, it is a table of
        # n_layers by n_experts counts built once when the model is profiled,
        # and it costs nothing in the decode loop. counted on the training rows
        # only, so it is scored on the same footing as the probes.
        static_scores = np.zeros_like(y, dtype=np.float32)
        for layer in range(n_layers_used):
            base = layer * n_per_layer
            counts = y[base + train.start : base + train.stop].sum(axis=0)
            static_scores[base : base + n_per_layer] = counts
        static = recall_at(static_scores[mask_test], y_te, budget)

        linear = recall_at(LinearProbe().fit(x_tr, y_tr).predict(x_te), y_te, budget)
        mlp = recall_at(
            MlpProbe(seed=seed).fit(x_tr, y_tr).predict(x_te), y_te, budget
        )

        results.append(LookaheadResult(k, budget, persistence, static, linear, mlp))
    return results
