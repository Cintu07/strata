"""traces with a known structure, for testing the harness itself.

this module exists so that every analysis in m0 can be exercised end to end
before a sixty gigabyte model has been downloaded, and so that each analysis can
be checked against a trace whose answer is known in advance. an analysis that
cannot recover a structure that was deliberately put there is broken, and it is
much cheaper to find that out here.

nothing produced here is a measurement. every trace is stamped
``SOURCE_SYNTHETIC`` and the report refuses to present one as a result.

the generator builds routing the way the design assumes a real model works, so
that the assumption is at least stated explicitly and can be argued with:

- each token has a latent state that drifts slowly and jumps at topic
  boundaries, which is where domain correlation comes from
- every layer routes by projecting that latent through its own fixed matrix,
  which is where multi-layer-ahead predictability comes from, since one latent
  drives every layer
- a per-layer expert bias makes some experts popular everywhere, which is the
  router load imbalance every paper reports
- with some probability a token simply repeats the previous token's choice,
  which is the persistence prior

``signal`` controls how much of the latent survives into the recorded hidden
state. at 1.0 the future is perfectly predictable from the present, at 0.0 the
hidden state is noise and the probe should recover nothing. both ends are worth
testing, because a probe that scores well on pure noise is measuring a leak.
"""

from __future__ import annotations

import numpy as np

from .trace import SOURCE_SYNTHETIC, Provenance, RouterTrace


def make_trace(
    n_tokens: int = 2000,
    n_layers: int = 12,
    n_experts: int = 32,
    top_k: int = 4,
    d_hidden: int = 64,
    *,
    n_domains: int = 4,
    domain_block: int = 250,
    persistence: float = 0.3,
    skew: float = 1.0,
    signal: float = 0.8,
    seed: int = 0,
    with_hidden: bool = True,
) -> RouterTrace:
    """generate a synthetic router trace.

    args:
        persistence: probability a token reuses the previous token's experts at
            a given layer. this is the free baseline the speculative router head
            has to beat.
        skew: strength of the per-layer expert popularity bias. zero is a
            perfectly balanced router, which no real one is.
        signal: how much of the latent survives into the recorded hidden state.
        with_hidden: whether to record hidden states at all. they dominate the
            file size, so an analysis that only needs routing can skip them.
    """
    rng = np.random.default_rng(seed)

    # per layer routing projection, shared across all tokens. one latent driving
    # every layer is what makes layer L+k predictable from layer L.
    proj = rng.normal(size=(n_layers, d_hidden, n_experts)).astype(np.float32)
    proj /= np.sqrt(d_hidden)
    bias = skew * rng.normal(size=(n_layers, n_experts)).astype(np.float32)

    domain_means = rng.normal(size=(n_domains, d_hidden)).astype(np.float32)

    latent = np.empty((n_tokens, d_hidden), dtype=np.float32)
    for t in range(n_tokens):
        domain = (t // domain_block) % n_domains
        # slow drift inside a topic, a jump when the topic changes
        latent[t] = domain_means[domain] + 0.35 * rng.normal(size=d_hidden).astype(np.float32)

    routing = np.empty((n_tokens, n_layers, top_k), dtype=np.int32)
    for layer in range(n_layers):
        scores = latent @ proj[layer] + bias[layer]
        choices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        routing[:, layer, :] = choices

    # the persistence prior, applied after the fact so that the recorded hidden
    # state does not explain the repeats. a probe that learns them is cheating.
    if persistence > 0:
        repeat = rng.random((n_tokens, n_layers)) < persistence
        repeat[0, :] = False
        for t in range(1, n_tokens):
            layers = np.flatnonzero(repeat[t])
            routing[t, layers, :] = routing[t - 1, layers, :]

    hidden = None
    if with_hidden:
        noise = rng.normal(size=(n_tokens, d_hidden)).astype(np.float32)
        observed = signal * latent + (1.0 - signal) * noise
        # every layer sees the same latent, with its own independent corruption,
        # so a probe cannot simply memorise one layer and reuse it
        per_layer_noise = rng.normal(size=(n_tokens, n_layers, d_hidden)).astype(np.float32)
        hidden = observed[:, None, :] + (1.0 - signal) * 0.5 * per_layer_noise

    return RouterTrace(
        routing=routing,
        n_experts=n_experts,
        hidden=hidden,
        provenance=Provenance(
            source=SOURCE_SYNTHETIC,
            model_id=f"synthetic-{n_layers}L-{n_experts}E-top{top_k}",
            corpus=f"generated, seed {seed}",
            notes=(
                f"persistence {persistence}, skew {skew}, signal {signal}, "
                f"{n_domains} domains in blocks of {domain_block}"
            ),
        ),
    )
