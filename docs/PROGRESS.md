# progress

last updated 2026-09-04.

## where the project is

**m2 and m4 are built and measured. m0's harness is complete and is waiting on a
real model.** the engine proper is still not started, and should not be until m0
answers.

## milestone status

| id | milestone | state |
|---|---|---|
| m0 | measurement harness | **harness complete and tested. not yet run against a real model**, which is the next action |
| m1 | correct reference implementation | not started |
| m2 | storage layer, io_uring plus O_DIRECT | **done and measured.** backend, alignment, deep queues, drain on drop, and a bandwidth benchmark |
| m3 | cache and eviction | **policy complete and measured against baselines.** not yet validated on real traces |
| m4 | expert-centric prefill | **done and measured.** bit identical to the reference, 68x fewer reads end to end |
| m5 | speculative router head | measurement side is in `m0/strata_m0/predict.py`, nothing shipped |
| m6 | hybrid cpu/gpu execution | not started |
| m7 | headline benchmark | not started |
| m8 | openai-compatible server | not started |

## what is done

**`crates/strata-format`** — the on-disk layout file. writer, reader, 128-byte
header, index with per-expert offsets, lengths, precisions and crc32s plus the
co-activation graph. every expert contiguous and 4kb aligned. positional reads
with no shared cursor. a read planner that coalesces a scattered want-set into a
few large transfers with three modes, including a per-expert baseline to measure
against. corruption in the header, the index or any single payload is caught
rather than fed to a gemm.

**`crates/strata-cache`** — a probationary window in front of a
greedy-dual-size-frequency main region, with admission decided by a 4-bit
count-min sketch with periodic halving. optional dequantised residency for the
hottest experts. lru and belady baselines in the same crate, driven through the
same interface.

**`crates/strata-layout`** — profile-guided placement. co-activation graph from
observed routing, pettis-hansen greedy chain merging per layer, and a capture
metric. verified end to end: an ordering written through the real file and read
back through the real planner turns four reads into one.

**`crates/strata-io`** — the storage tier. `io_uring` with `O_DIRECT` on linux,
positional reads everywhere else, behind one submit-then-wait interface. an
aligned slot pool, backpressure rather than blocking, short reads reported
against the op that caused them, and a drain in `Drop` so the kernel can never
write into freed memory. byte parity against the portable reference is a test.

**`crates/strata-prefill`** — expert-centric prefill. inverts token-to-expert
into expert-to-token, orders batches by disk offset, and executes them so that
each expert is read once per layer rather than once per token. **bit identical**
to the token-major reference, not merely close, for the reason in decision 0010.

**`crates/strata-bench`** — the end to end measurement, and `strata-io`'s
`bandwidth` binary.

**`m0/`** — the measurement harness. trace schema with provenance, router hooks
for hugging face moe models, a synthetic generator for testing the harness
itself, and five analyses. writes figures, a markdown report with a go/no-go
verdict table, and a machine-readable summary.

## what is deliberately not done

- **no gguf or safetensors converter.** the layout writer takes bytes; nothing
  yet turns a real model file into those bytes. this is the next piece of
  plumbing after m0 reports.
- **no engine.** no attention, no kernels, no decode loop, no server.
- **no prefetcher.** m5's fallback ladder is designed and not built, because
  what it should do depends on what m0 measures.
- **adaptive cache window is implemented but off.** see decision 0006.

## next action

run m0 against a real mixture-of-experts model.

`ibm-granite/granite-3.1-1b-a400m-instruct` is a good first target: 24 layers,
32 experts, top-8, about 2.7gb, so 768 expert-layer pairs and a download that
finishes in minutes rather than hours. `allenai/OLMoE-1B-7B-0924` is the better
second one at 16 layers and 64 experts, and is 13.8gb.

the harness needs a corpus of real prompts from a workload worth caring about,
not filler. routing is domain correlated, which is the entire reason the cache
policy is built the way it is, so a corpus of lorem ipsum measures the wrong
distribution and every number downstream inherits that.

then publish the m0 numbers as a standalone writeup **before** building the
engine, as appendix b of the prd says. nobody has published these cleanly.

## measurements on record

all of these are reproducible from the repo. none of them is a statement about a
real model.

### storage, `cargo run --release -p strata-io --bin bandwidth`

```
pattern      block   qd      GB/s        IOPS     lat us
sequential      1M    1     1.737        1656      603.8
sequential      1M   16     3.514        3352     4774.0
random          4K    1     0.023        5495      182.0
random          4K  128     0.406       99181     1290.6
random         64K   16     2.496       38089      420.1
random          1M    4     3.843        3665     1091.5
```

queue depth is worth 18x on random 4k. sequential is worth 156x over random 4k
at queue depth one. and **random 1M matches sequential 1M**, which is the result
that reframed what the layout is for. see decision 0009.

measured inside wsl2, so an ext4 volume on a virtual disk on ntfs. the shape is
real, the peak is a lower bound.

### prefill, `cargo run --release -p strata-bench --bin prefill`

1024 tokens, 4 layers, top-8 of 128 experts, 2 MiB each.

```
stage             reads   bytes read      GB/s    time ms
token-major       32768       65536 M      2.78    24685.9   extrapolated from 12%
expert-major        482         964 M      3.47      291.4   measured in full
+ coalesced          43         964 M      3.37      300.2   measured in full
+ warm cache        373         746 M      3.27      239.4   measured in full
```

68x fewer reads, 68x fewer bytes, about 85x less time. coalescing merges 482
reads into 43 and saves no time at all, because 2 MiB reads are already past the
size where request count matters. that is the point of decision 0009, and it
says coalescing earns its keep on small experts, not large ones.

### cache, `cargo test -p strata-cache --test measure -- --nocapture`

hit rate by policy, capacity in experts:

```
hot set of 16 revisited, 4 one shot experts per round
  cap  16  lru 0.562   strata-no-admission 0.668   strata 0.762   oracle 0.783

four domains of 12 experts, 300 accesses per block, 24 blocks
  cap  24  lru 0.960   strata-no-admission 0.946   strata 0.891   oracle 0.972

skewed access over 64 experts in one layer
  cap  24  lru 0.617   strata-no-admission 0.671   strata 0.716   oracle 0.835
```

strata leads on scan traffic and on skew, and trails lru on the pure-recency
workload. that last row is a known trade, asserted as a test, and the ablation
shows the cost is admission specifically.
