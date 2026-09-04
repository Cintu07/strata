# m0: the measurement harness

the go/no-go gate for the whole project. it instruments a real mixture-of-experts model and answers the questions strata's design assumes the answers to.

**this is not a warm-up milestone.** if the numbers say stop, the deliverable is a short negative-result writeup and the engine does not get built. three weeks and a clean set of numbers nobody has published is a contribution.

## the five questions

| # | question | what a bad answer means |
|---|---|---|
| 1 | do consecutive tokens reuse experts | if routing is effectively random, nothing can be cached and no policy work helps |
| 2 | how skewed is router load | a uniform router pins the hit rate to the ram ratio |
| 3 | what hit rate is achievable at a given ram budget | this is the curve that tells a user how much ram their model needs |
| 4 | do experts fire together in stable groups | without co-activation structure, laying the file out by it buys nothing |
| 5 | **can layer L+k routing be predicted from layer L's hidden state** | if not, multi-layer-ahead prefetch does not exist and the research contribution is gone |

question five is the one that can kill the project. nvme read latency is 50 to 150 microseconds and a layer of laptop compute can be shorter than that, so one layer of lookahead cannot hide a read. three to eight layers can. nobody has built that, because in a ram-resident world nobody needed to.

## install

```bash
pip install -e ".[dev]"            # numpy, matplotlib, pytest
pip install -e ".[dev,capture]"    # adds torch and transformers, only needed to capture
```

## use

```bash
# exercise the entire pipeline with no model download
python -m strata_m0 synth   --out traces/dev.npz
python -m strata_m0 analyse --trace traces/dev.npz --out results/dev

# the real thing
python -m strata_m0 capture --model <hf-moe-id> --corpus corpus.txt --hidden --out traces/real.npz
python -m strata_m0 analyse --trace traces/real.npz --out results/real
```

`analyse` writes `report.md` with a verdict table, five figures, and `summary.json` for anything downstream.

## provenance is enforced

every trace records whether it was captured or generated, and the report puts a refusal banner at the top of anything built from a synthetic one:

> **this is not a measurement.** the numbers below describe a trace that was generated with the structure the design assumes, so of course it exhibits that structure.

a synthetic trace and a captured one have exactly the same shape, which is precisely why the distinction has to be carried in the data and not in the reader's memory.

## what the synthetic generator is for

testing the harness, not producing results. it builds routing the way the design assumes a real model works, and stating that assumption explicitly is half the value:

- each token has a latent state that drifts slowly and jumps at topic boundaries, which is where domain correlation comes from
- every layer routes by projecting that same latent, which is where multi-layer-ahead predictability comes from
- a per-layer expert bias produces router load imbalance
- with some probability a token repeats the previous token's choice, which is the persistence prior

`signal` controls how much of the latent survives into the recorded hidden state. `tests/test_harness.py` checks both ends: at `signal=1.0` a probe must recover the structure, and at `signal=0.0` it must find nothing. a probe that scores well on pure noise is measuring a leak in the evaluation, and that is much cheaper to catch here than after a capture run.

## on the corpus

a trace is only as representative as the text it was captured on. use real prompts from the workload you care about. routing is domain-correlated, which is the entire reason the cache policy is built the way it is, so a corpus of filler measures the wrong distribution and every downstream number inherits that.

## on capture cost

a hidden state is one vector per token per layer. at 4096 dimensions across 60 layers that is about a megabyte per token, which is unusable. capture projects down to `--probe-dim` with a fixed seeded random projection, which preserves the geometry a probe cares about and turns the file into something you can keep.

## why the baselines are here

the persistence prior costs nothing: assume token t routes at layer L+k the way token t-1 did. it needs no model, no training and no inference. any speculative router head that does not beat it should not be shipped, so it is measured alongside every probe rather than mentioned in passing.

the same reasoning puts lru and belady in the cache sweep. lru is the floor every system in this space reports against; belady is the ceiling that separates a bad policy from a cache that is simply too small.
