# progress

last updated 2026-09-04.

## where the project is

**pre-m0.** the foundation the measurement harness and the storage design need is built and tested. the engine is not started, and should not be until m0 answers.

## milestone status

| id | milestone | state |
|---|---|---|
| m0 | measurement harness | **harness complete, not yet run against a real model.** this is the next action |
| m1 | correct reference implementation | not started |
| m2 | storage layer, io_uring plus O_DIRECT | **format and read planner done, direct-io backend not started** |
| m3 | cache and eviction | **policy complete and measured against baselines, not yet validated on real traces** |
| m4 | expert-centric prefill | not started |
| m5 | speculative router head | measurement side is in `m0/strata_m0/predict.py`, nothing shipped |
| m6 | hybrid cpu/gpu execution | not started |
| m7 | headline benchmark | not started |
| m8 | openai-compatible server | not started |

## what is done

**`crates/strata-format`** — the on-disk layout file. writer, reader, 128-byte header, index with per-expert offsets, lengths, precisions and crc32s plus the co-activation graph. every expert contiguous and 4kb-aligned. positional reads with no shared cursor, so the io path is ready for many concurrent transfers. a read planner that coalesces a scattered want-set into a few large sequential transfers and reports exactly what the coalescing cost, with three modes: default bridging, `no_overfetch`, and `per_expert` as the measurement baseline. corruption in the header, the index or any single payload is caught rather than fed to a gemm. 32 tests.

**`crates/strata-cache`** — the expert cache. a probationary lru window in front of a greedy-dual-size-frequency main region, with admission decided by a 4-bit count-min sketch with periodic halving. optional dequantised residency for the hottest experts. lru and belady baselines in the same crate driven through the same interface. 22 tests plus a measurement binary.

**`crates/strata-layout`** — profile-guided placement. accumulates a co-activation graph from observed routing, orders each layer with pettis-hansen greedy chain merging, and reports how much co-activation weight an order actually captures. verified end to end: an ordering produced from a profile, written through the real layout file and read back through the real planner, turns four reads into one. 10 tests.

**`m0/`** — the measurement harness. trace schema with provenance, router hooks for hugging face moe models with random-projection compression of hidden states, a synthetic generator for testing the harness itself, and five analyses: reuse across tokens, access skew, hit rate against cache size with lru/lfu/belady, co-activation structure, and multi-layer-ahead predictability with linear and mlp probes against the persistence prior. writes figures, a markdown report with a go/no-go verdict table, and a machine-readable summary. 26 tests.

## what is deliberately not done

- **no io_uring or O_DIRECT backend.** the format, the alignment, and the planner all exist for it, but writing an async direct-io path that cannot be compile-tested on this machine would be shipping untested code that claims to work. m2.
- **no gguf or safetensors converter.** the layout writer takes bytes; nothing yet produces those bytes from a real model file.
- **no engine.** no attention, no kernels, no scheduler, no server.
- **adaptive cache window is implemented but off.** see decision 0006.

## next action

run m0 against a real mixture-of-experts model. a 30B-class model with roughly 3B active is the right size to iterate on. the harness needs a corpus of real prompts from a workload worth caring about, not filler, because a trace is only as representative as the text it was captured on.

then publish the m0 numbers as a standalone writeup **before** building the engine, as appendix b of the prd says. nobody has published these cleanly.

## measurements on record

from `cargo test -p strata-cache --test measure -- --nocapture`, hit rate by policy, capacity in experts:

```
hot set of 16 revisited, 4 one shot experts per round
  cap  16  lru 0.562   strata-no-admission 0.668   strata 0.762   oracle 0.783

four domains of 12 experts, 300 accesses per block, 24 blocks
  cap  24  lru 0.960   strata-no-admission 0.946   strata 0.891   oracle 0.972

skewed access over 64 experts in one layer
  cap  24  lru 0.617   strata-no-admission 0.671   strata 0.716   oracle 0.835
```

strata leads on scan traffic and on skew, and trails lru on the pure-recency workload. that last row is a known trade, asserted as a test, and the ablation shows the cost is admission specifically.

these are synthetic workloads and none of them is a result about a real model. they exist to check that the policy responds to a known structure the way the design says it should.
