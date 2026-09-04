# strata

**a memory-hierarchy-native inference engine for frontier mixture-of-experts models on ram-constrained consumer hardware**

strata targets the genuinely ram-constrained regime: a 200B+ parameter moe model on a 16GB laptop, where nvme is on the steady-state critical path of every token rather than in the cold-start path. its two mechanisms are multi-layer-ahead expert prediction and expert-centric prefill scheduling.

that positioning is deliberate and narrow. ktransformers, freetoken, llama.cpp and the expert-offloading literature all assume the hot working set eventually fits in ram and treat disk as cold start or as a prefix cache. nobody has built for the machine most people actually own. see [docs/PRD.md](docs/PRD.md) section 3 for the competitive landscape and what it leaves open.

## status

**pre-m0.** the engine does not exist yet, and it should not until m0 says it is worth building.

what is here is the foundation m0 and the storage design need, all of it tested and none of it stubbed:

| crate | what it does |
|---|---|
| [`strata-format`](crates/strata-format) | the on-disk expert layout file: 4kb-aligned contiguous experts, an index carrying offsets, checksums, per-expert precision and the co-activation graph, and a read planner that turns a scattered want-set into a few large sequential transfers |
| [`strata-cache`](crates/strata-cache) | the expert cache: a probationary window in front of a greedy-dual-size-frequency main region with tinylfu admission, plus the lru and belady baselines it is measured against |
| [`strata-layout`](crates/strata-layout) | the profile-guided placement pass: greedy chain merging over the measured co-activation graph, so experts that fire together are neighbours on disk |
| [`m0/`](m0) | the measurement harness: the go/no-go gate for the entire project |

no external dependencies in any of the three crates. `cargo test` works offline.

## the go/no-go gate

m0 is not a warm-up, it is the falsification test. it instruments a real moe model and answers five questions, any one of which can end the project:

1. do consecutive tokens reuse experts, or is routing effectively random
2. how skewed is router load
3. what hit rate is achievable at a given ram budget, and where is the knee
4. do experts fire together in stable groups
5. **can layer L+k routing be predicted from layer L's hidden state**

question five is the one the research contribution rests on. nvme read latency is 50 to 150 microseconds and a layer of laptop compute can be shorter than that, so one layer of lookahead cannot hide a read. three to eight layers can, and nobody has built that because in a ram-resident world nobody needed to.

if the answers say stop, the right outcome is a short negative-result writeup. three weeks and a clean set of numbers nobody has published is a contribution. a year spent on a false premise is not.

```bash
cd m0
pip install -e ".[dev]"                  # add [capture] for torch and transformers

# exercise the whole pipeline with no model download
python -m strata_m0 synth   --out traces/dev.npz
python -m strata_m0 analyse --trace traces/dev.npz --out results/dev

# the real thing
python -m strata_m0 capture --model <hf-moe-id> --corpus corpus.txt --hidden --out traces/real.npz
python -m strata_m0 analyse --trace traces/real.npz --out results/real
```

the report stamps its own provenance and **refuses to present a synthetic trace as a result**. a synthetic trace was built to contain the structure the design assumes, so of course it contains it.

## running the tests

```bash
./scripts/test.sh
```

on windows this machine has no msvc linker, so cargo runs through wsl:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/Pawan/Desktop/strata && ./scripts/test.sh"
```

## two things the measurements already changed

the point of building measurement in first is that it talks back. two examples from this repo:

**the cache policy deadlocked, and the comparison table caught it.** the first version scored resident entries by a lifetime access count and admitted a newcomer only if it outscored the entry it would evict. after one topic runs for a while its experts carry counts in the tens, every newcomer arrives with one, so nothing is admitted, so nothing is evicted, so the aging clock never advances and the cache freezes on whatever it saw first. it scored 0.25 against lru's 0.96. the fix is tinylfu: compare both sides using a sketch over recent accesses, misses included, so an expert returning after a digression has the history to get back in. see [`sketch.rs`](crates/strata-cache/src/sketch.rs).

**adaptive window sizing was measured and switched off.** caffeine's hill-climbing window is a sound idea and it lost ground almost everywhere it moved on these workloads. tuning an adaptive controller against synthetic traces is fitting noise, so it stays off until m0 produces real ones. the code and the flag are still there.

`cargo test -p strata-cache --test measure -- --nocapture` prints the table those decisions came from.

## where the honest limits are

- **strata loses to lru on pure short-range recency.** a workload where a topic runs long enough to turn the whole cache over, with no reuse across topics, is lru's best case, and admission control is a cost there. this is asserted as a test rather than left out of the readme: see `a_pure_recency_workload_is_where_lru_still_wins`.
- **belady is exactly optimal only for uniform expert sizes.** across layers of differing width it is a close estimate of the ceiling, not a proof.
- **no io_uring backend yet.** the format and the read planner are built for it and the reader uses positional reads with no shared cursor, but m2 is where the direct-io path gets written and where achieved bandwidth gets verified against device spec.

## documents

- [docs/PRD.md](docs/PRD.md) — the full product requirements document
- [docs/PROGRESS.md](docs/PROGRESS.md) — what is done, what is next, what is deliberately not started
- [docs/decisions/](docs/decisions) — the decisions that would otherwise get re-litigated

## licence

apache-2.0.
