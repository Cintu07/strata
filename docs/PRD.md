# strata

**a memory-hierarchy-native inference engine for frontier mixture-of-experts models on ram-constrained consumer hardware**

product requirements document · v1.0 · september 2026

---

## 1. the one sentence

strata runs a 200B+ parameter mixture-of-experts model on a 16GB laptop at usable speed, by treating nvme as a first-class tier of the memory hierarchy rather than as a place the model is loaded from.

---

## 2. the problem

### 2.1 the gap that makes people give up their data

there are two tiers of local inference today:

- **models that fit in ram.** 8B to 30B. these run well. llama.cpp, ollama and mlx solved this.
- **models worth using for hard work.** 200B to 1T. these need a workstation or an api.

the space between them is where almost everyone actually lives. a person with a 16GB laptop who wants frontier-class reasoning has exactly one option, which is to send their data to someone else's server. for a lawyer, a clinician, a journalist protecting a source, or an engineer under an nda, that option is not available, so they simply do not get the capability at all.

closing this gap is not a performance optimisation. it changes who has access.

### 2.2 why moe makes it possible

dense models are hopeless here. every weight is read for every token, so a 405B dense model must move 405B parameters per token regardless of what you do.

moe models break that. only a small fraction of parameters are active per token:

| model | total params | active per token | ratio |
|---|---|---|---|
| qwen3-30b-a3b | 30B | 3B | 10% |
| gpt-oss-120b | 117B | ~5.1B | 4.4% |
| deepseek-v3 class | 671B | 37B | 5.5% |
| deepseek-v4-flash class | 284B | small | low single digits |
| kimi k2 class | ~1T | 32B | 3.2% |

the structure is what matters. each layer has:

- **hot weights**: attention, layernorms, router, embeddings, shared experts. always needed, small.
- **cold weights**: routed expert ffns. enormous in total, and only top-k of N are touched per token.

so the model is 95%+ dormant at any instant. the only question is whether you can get the right 5% into place fast enough.

### 2.3 the bandwidth arithmetic, stated honestly

single-stream decode is bandwidth bound. the ceiling is:

```
tokens/sec = effective bandwidth / bytes that must move per token
```

| tier | bandwidth | latency |
|---|---|---|
| gpu vram | 300 GB/s to 1 TB/s | ~100 ns |
| system ram (lpddr5) | 60 to 130 GB/s | ~100 ns |
| pcie 4 x16 | ~26 GB/s | ~1 us |
| nvme gen4, sequential | 5 to 7 GB/s | 50 to 100 us |
| nvme gen4, random 4K, qd1 | **under 0.1 GB/s** | 80 to 150 us |

**the last row is the entire engineering problem.** nvme sequential bandwidth looks survivable. nvme random small-read bandwidth is catastrophic. every design decision in this document exists to convert random access into sequential access, and to hide the latency that remains.

worked example. gpt-oss-120b at 4-bit is roughly 60GB on disk. hot weights are roughly 4GB and stay resident. per token, expert reads are roughly 1.5GB if nothing is cached.

- naive, no cache, random reads: unusable, well under 1 tok/s.
- naive, no cache, perfect sequential reads at 6 GB/s: 4 tok/s.
- with a 70% expert cache hit rate: 0.45GB from disk per token, roughly 13 tok/s.
- with 70% hit rate and full prefetch overlap: bounded by compute instead, roughly 20+ tok/s.

**the cache hit rate is the product.** everything else is plumbing around it.

---

## 3. the competitive landscape, stated honestly

this is not an empty field. do not start until you have read this section and accepted its implications.

| system | what it does | what it assumes | gap it leaves |
|---|---|---|---|
| **ktransformers** (sosp 2025) | cpu/gpu hybrid moe, experts in system ram, static offload rules, gpu-cpu-disk prefix cache | **large system ram**, often 200GB+ workstation class | does not solve the case where experts cannot fit in ram at all |
| **freetoken** (berkeley + mit, aug 2026) | dynamic per-layer cpu/gpu splits computed in closed form at runtime; reports ~39 tok/s for a 35B moe on an 8GB rtx 4060 laptop, and very large models on workstation gpus | still fundamentally weights-in-ram with gpu as the scarce tier | the ram-constrained regime, where nvme is in the steady-state critical path rather than the cold-start path |
| **llama.cpp / ollama** | gguf, layer-wise offload, mmap; an ssd expert-streaming proposal is under discussion, and a metal proof of concept reports ~13 tok/s on an m1 pro 16GB with lru expert paging | mmap and os page cache do the memory management | no expert-aware prefetch, no prediction, no expert-centric scheduling, and mmap is the wrong primitive (see 5.2) |
| **hobbit** | mixed precision expert offloading | | partially occupies the precision-by-frequency idea |
| **promoe** | proactive expert caching | | occupies single-layer-ahead prediction |
| **pipo** | pipelined offloading for consumer devices | | pipeline overlap, but not the prediction horizon problem |

### 3.1 what is therefore actually open

three things, in order of value:

**o1. the genuinely ram-constrained regime.** every system above assumes the hot working set eventually fits in ram, and treats disk as cold start or as a prefix cache. nobody has built for the case where **nvme is on the steady-state critical path of every token**. that is the 16GB laptop with a 200B model, which is the machine most people actually own.

**o2. multi-layer-ahead expert prediction.** existing proactive caching predicts roughly one layer ahead. on a laptop, nvme read latency is 50 to 150 us and a layer of compute may take less than that, so **one layer of lookahead is not enough to hide the read**. you need a prediction horizon of 3 to 8 layers. nobody has built that, because in a ram-resident world you never needed it.

**o3. expert-centric prefill scheduling.** during prefill you have all tokens at once, so you can reorder the computation to group tokens by expert, load each expert exactly once, and apply it to every token that routed to it. this converts prefill from random access into a single sequential sweep of the expert file. the large prefill speedups reported by others suggest the headroom is real and not yet exhausted.

**strata is the system that takes o1 as its target and o2 and o3 as its mechanisms.** that is the whole positioning, and it should appear in the first paragraph of the readme.

---

## 4. goals and non-goals

### 4.1 goals

| # | goal |
|---|---|
| g1 | run a 200B+ parameter moe model on a machine with 16GB ram and no discrete gpu, at 10+ tok/s decode |
| g2 | achieve an expert cache hit rate above 70% on realistic multi-turn workloads |
| g3 | hide nvme latency behind compute, so measured decode throughput is within 2x of the no-io ceiling |
| g4 | prefill that reads each needed expert from disk at most once per prefill, not once per token |
| g5 | correctness identical to a fully resident reference implementation, verifiable by logit diff |
| g6 | graceful degradation: as ram shrinks, throughput should fall smoothly rather than off a cliff |

### 4.2 non-goals

- **not** a training or fine-tuning system.
- **not** a datacenter server. no continuous batching, no multi-tenant scheduling. those assume high interconnect bandwidth and are already well served by vllm and sglang.
- **not** a dense model runtime. dense models cannot be helped by any of this and pretending otherwise dilutes the pitch.
- **not** a new quantization format. consume gguf and safetensors; do not invent a competitor.
- **not** a chat ui.

---

## 5. architecture

### 5.1 the tier model

```
  TIER 0   GPU VRAM (if present)
           hot weights, kv cache, top-N hottest experts
           residency decided by measured access frequency

  TIER 1   SYSTEM RAM
           hot weights (always), warm expert cache, prefetch staging buffers
           this is the tier the cache policy actually manages

  TIER 2   NVME
           the complete expert set, laid out for sequential access
           read via io_uring with O_DIRECT
```

on unified-memory machines (apple silicon), tiers 0 and 1 collapse into one, which simplifies the policy and changes the tuning constants. treat it as a supported configuration, not an afterthought.

### 5.2 storage layer

**mmap is the wrong primitive and using it is the single most common mistake in this space.**

- page faults are synchronous, so the thread stalls with no way to overlap
- 4KB granularity fights you when experts are 10 to 100MB
- the os page cache duplicates data you are already caching, wasting the ram you are short of
- eviction policy is the kernel's, and the kernel does not know what a router is
- fault latency is invisible to your scheduler, so you cannot reason about it

**instead: io_uring with O_DIRECT.**

- deep queues, so many expert reads are in flight simultaneously; this is the only way to get real bandwidth from nvme, since queue depth 1 random reads are catastrophically slow
- completion driven, so the scheduler knows exactly when a weight has arrived
- bypasses the page cache, so your ram budget is yours
- explicit alignment, which you need for direct io anyway

**on-disk layout.** the file format is part of the product, not a detail.

- each expert stored contiguously and 4KB aligned, so one expert is one large sequential read
- experts co-located by **measured co-activation**, not by index. if experts 3 and 47 in layer 12 fire together often, put them adjacent so one read fetches both. this is a profile-guided layout pass, run once per model, and it is a real differentiator: it converts a class of cache misses into free bytes on an existing read.
- a small header with per-expert offsets, sizes, and a co-activation graph, so the runtime can plan reads without scanning
- optional per-expert precision, so hot experts can be stored at higher precision than cold ones

### 5.3 the cache

**what to cache.** not experts. **expert-layer pairs.** expert 5 in layer 3 and expert 5 in layer 30 are unrelated objects and must be tracked separately.

**why lru is wrong.** expert access is heavily skewed. router load is well known to be imbalanced, and access patterns are also domain-correlated: a coding conversation hits a stable subset of experts, and a different subset dominates for a different domain. pure recency throws away that structure on every topic switch.

**the policy: cost-aware frequency with recency decay.**

```
score(e) = (decayed access frequency of e)
         × (cost to reload e = bytes / achievable sequential bandwidth)
         ÷ (bytes resident for e)
```

evict lowest score. the cost term matters because experts differ in size across models, and the byte term makes it a proper knapsack rather than a popularity contest.

**admission control.** do not cache on first touch. an expert seen once in a long context is probably noise. admit on second access within a window, which is the standard fix for cache pollution from one-shot accesses and applies cleanly here.

**segmented residency.** for very hot experts, keep them dequantized in ram. for warm ones, keep them quantized and dequantize on use, which trades a little compute for a lot of capacity. this is a per-expert decision driven by the same frequency statistics.

### 5.4 prediction and prefetch, the core contribution

this is o2 from section 3.1 and it is where the project earns its existence.

**the problem.** layer L+1's router needs layer L's output, so the true expert set is not known until layer L finishes. by then, an nvme read is too late.

**the horizon requirement.** to hide a 100 us read behind compute, you need enough lookahead that the read is issued at least one read-latency before use. on a laptop that means **3 to 8 layers of lookahead**, not one. this is the specific thing existing proactive caching systems do not do, and it is not a small extension of them.

**the mechanism: a speculative router head.**

a tiny mlp, on the order of a few hundred kilobytes, trained offline on traces from the target model. it takes the hidden state at layer L and predicts the expert sets for layers L+1 through L+k. it is trained once per model and shipped alongside it.

three properties make this tractable:

1. **it only needs to be right about the union, not the exact set.** prefetching a superset is fine; you waste a little bandwidth and lose nothing else.
2. **being wrong is never a correctness issue.** this is prefetch, not speculative execution. a miss costs a stall, and the true router still runs and still decides. there is no rollback and no output change.
3. **it can be wrong asymmetrically and still win.** optimise for recall over precision, because a false positive costs bandwidth and a false negative costs a full stall.

**why this should work at all.** routing is not random. it correlates strongly with token identity, with position, and with the domain of the conversation. an expert set that is stable across a paragraph of code is predictable from the hidden state well before the router computes it. **whether this holds at the accuracy needed is the central empirical question of the project, and m0 exists to answer it before anything else is built.**

**fallback ladder**, in order of cheapness, all of which should ship:

1. **persistence prior.** the same expert selected at layer L for token t is likely to be selected at layer L for token t+1. free, requires no model, and is a strong baseline you must beat.
2. **co-activation prefetch.** on a miss, fetch the missed expert plus its top co-activation neighbours, since they are adjacent on disk and therefore nearly free.
3. **speculative router head.** the real mechanism.
4. **full-layer fetch under low load.** if the io queue is idle, fetch everything for the next layer. bandwidth spent while idle costs nothing.

### 5.5 expert-centric prefill

this is o3, and it is the largest single win in the system.

**the standard order** is token-major: for each token, for each layer, route and compute. every token independently demands its experts, so an expert may be read many times in one prefill.

**the strata order** is expert-major, per layer:

```
for each layer L:
    run attention for all tokens (hot weights, always resident)
    run the router for all tokens, producing token -> expert assignments
    invert to expert -> token list
    sort experts by disk offset          # sequential sweep, not random access
    for each expert in that order:
        issue read (deep io_uring queue, many in flight)
        on arrival, apply to all its assigned tokens as one batched gemm
    combine outputs back into token order
```

this achieves three things at once:

- **each expert is read at most once per layer per prefill**, satisfying g4
- **reads are issued in disk order**, so a random access pattern becomes one sequential sweep of the file
- **the ffn becomes a batched gemm over many tokens**, which raises arithmetic intensity and makes the compute efficient too

the cost is peak memory for intermediate activations, which is bounded by chunking the prefill into token blocks sized to the ram budget.

### 5.6 hybrid execution, the inversion

when an expert's weights are in system ram and the compute device is a gpu, the obvious move is to copy the weights to the gpu. **for decode, that is usually wrong.**

decode touches an expert with very few tokens. the weight is tens of megabytes; the activation is a few kilobytes.

```
transfer weights to gpu:  ~50 MB over pcie   ≈ 2 ms
compute the expert on cpu: a few tokens of ffn ≈ 0.5 ms on avx-512
```

so **move the activation to the weights, not the weights to the activation.** compute in ram-resident experts on the cpu, and reserve the gpu for hot weights, attention, and the experts hot enough to be permanently vram-resident.

the crossover is a function of token count per expert, weight size, pcie bandwidth and cpu throughput, and it flips during prefill where token counts are large. **compute the crossover at runtime from measured constants rather than hardcoding a rule.** this is the axis on which prior static-rule systems are weakest and where dynamic-split systems have shown real gains, so match them here or you will lose on this axis alone.

### 5.7 stack

| layer | choice | reason |
|---|---|---|
| core | rust | no gc pauses in the token loop; ownership maps cleanly onto tiered buffer lifetimes and pinned io buffers |
| storage | io_uring (linux), io_ring (windows), kqueue plus aio (macos) | async, deep queue, direct io |
| cpu kernels | avx-512 / avx2, and neon on apple silicon | cpu expert execution is a first-class path, not a fallback |
| gpu | cuda first, then metal, then vulkan | metal matters more than usual here, because apple laptops have the bandwidth and the unified memory |
| formats | read gguf and safetensors; emit a strata layout file | consume the ecosystem, do not fight it |
| api | openai-compatible http server plus a rust library | adoption requires being a drop-in replacement |

---

## 6. milestones

| id | milestone | done when |
|---|---|---|
| **m0** | **the measurement harness** | instrument a real moe model and publish: expert reuse rate across consecutive tokens, cache hit rate under an oracle of size X, expert access skew, co-activation structure, and the predictability of layer L+k routing from layer L's hidden state. **this is the go/no-go gate for the entire project.** |
| **m1** | correct reference | a fully resident implementation whose logits match huggingface within tolerance; every later optimisation is diffed against this |
| **m2** | storage layer | io_uring plus O_DIRECT, strata layout file, profile-guided co-activation ordering; measured sequential read bandwidth above 80% of device spec |
| **m3** | cache and eviction | cost-aware frequency policy with admission control; hit rate above 70% on multi-turn workloads (g2) |
| **m4** | **expert-centric prefill** | each expert read at most once per layer per prefill; prefill throughput compared against llama.cpp and freetoken |
| **m5** | **speculative router head** | multi-layer-ahead prediction with recall above 90% at k=4; measured stall reduction, not just prediction accuracy |
| **m6** | hybrid cpu/gpu execution | runtime crossover computation; correct decisions verified against exhaustive measurement on two machines |
| **m7** | the headline benchmark | a 200B+ moe model, 16GB ram machine, 10+ tok/s decode, reproducible, with a scripted harness anyone can run |
| **m8** | openai-compatible server | it is a drop-in replacement for ollama in an existing application |

**m0 is not a warm-up. it is the project's falsification test.** if expert reuse across tokens is low, or if layer L+4 routing is not predictable from layer L, strata cannot work as designed and you should publish the measurements as a short negative-result writeup and stop. that outcome costs three weeks and is a genuine contribution, because nobody has published these numbers cleanly.

---

## 7. benchmarks

### 7.1 hardware, all three tiers

| class | spec | what it tests |
|---|---|---|
| **the target machine** | 16GB ram, no discrete gpu, gen4 nvme | g1. this is the machine the project exists for |
| mid laptop | 32GB ram, 8GB vram, gen4 nvme | the common developer machine |
| apple silicon | m-series, 16GB unified | the unified memory path |

### 7.2 models

one small moe for iteration speed (a 30B-class model with roughly 3B active), one mid (a 120B-class model), one large (a 280B+ class model). the large one is the headline.

### 7.3 baselines

llama.cpp with expert offload, ollama, ktransformers, and freetoken. **run them yourself on your hardware.** do not quote their published numbers against your own measurements; different machines make that comparison meaningless and reviewers will say so immediately.

### 7.4 metrics

- decode tok/s, and time to first token
- **bytes read from nvme per token.** the metric that most directly reflects the actual contribution, and the one nobody reports
- expert cache hit rate
- prediction recall at k layers ahead
- achieved nvme bandwidth as a fraction of device spec
- peak rss, so the ram claim is verifiable
- joules per token, measured with rapl or powermetrics
- **sustained thirty-minute throughput**, which is where thermal throttling shows up and where every short benchmark lies

---

## 8. the money graph

**x-axis:** model total parameters, log scale, from 30B to 1T.
**y-axis:** decode tokens per second, log scale.
**annotation:** a vertical line marking where the model exceeds the machine's ram.

llama.cpp and ollama track along fine and then **fall off a cliff** at that line, because past it they are thrashing the page cache.

strata continues past the line with a **gentle downward slope**, because past it the system is doing exactly what it was designed to do.

the whole pitch is the shape of two curves at one vertical line. it needs no caption.

**a second figure worth having:** cache hit rate against ram budget, showing the knee. it tells a buyer exactly how much ram they need for their model, which is the question every actual user has.

---

## 9. risks

| risk | severity | mitigation |
|---|---|---|
| **expert reuse across tokens is too low, so caching cannot work** | **existential** | m0 measures this before anything is built. this is the entire reason m0 is first |
| **multi-layer-ahead routing is not predictable** | **existential** | also measured at m0. if recall at k=4 is poor, fall back to the persistence prior plus co-activation prefetch and reposition the project around o1 and o3 alone, which are still valuable |
| the field moves faster than you build | **high** | this is a race, not an empty field. pick o1 and stay narrow. a system that is the best in the world at the 16GB case beats a system that is fourth-best at everything |
| consumer nvme random read at low queue depth is far worse than spec | high | the entire storage design exists to avoid this. verify achieved bandwidth at m2 before building on top |
| thermal throttling on sustained laptop nvme and cpu load | medium | the thirty-minute sustained benchmark is mandatory, not optional. most published laptop numbers are short-burst and quietly dishonest |
| os page cache fights you for ram | medium | O_DIRECT, which is a reason to avoid mmap entirely |
| moe architectures change and break your assumptions | medium | shared experts, fine-grained experts and varying top-k already differ across model families. abstract the router interface from day one |
| nobody adopts it | high | openai-compatible api at m8. being a drop-in replacement is the adoption strategy; asking people to rewrite their integration is asking them not to switch |

---

## 10. what makes it legible in sixty seconds

1. **the money graph.** two curves, one vertical line. requires no explanation.
2. **a screen recording.** `htop` showing 14GB of ram used, next to a terminal streaming coherent output from a 284B model, with a tok/s counter running. the contradiction between those two numbers is the entire pitch.
3. **one line in the readme:** *"a 284B model. a 16GB laptop. 12 tokens per second. no cloud."*

---

## 11. why it matters

**it changes who has access.** the people locked out of frontier models are not people who dislike apis. they are people who cannot legally or practically send their data anywhere: clinicians, lawyers, journalists protecting sources, engineers under nda, and everyone in a jurisdiction where the data cannot leave. for them the choice today is not "local versus cloud", it is "frontier capability versus none".

**it changes the cost floor.** a capable local model on hardware people already own removes a recurring cost from every student, indie developer and small team that currently cannot justify one. this matters most in exactly the places where gpu access is hardest to buy.

**it makes agentic workloads viable locally.** long autonomous runs care about total throughput and total cost, not about the latency of any single token. a laptop that grinds overnight at 12 tok/s is genuinely useful for that, and costs nothing per token.

**it is a durable engineering position.** models keep getting sparser and total parameter counts keep rising, which widens the gap between total and active parameters every year. the ratio moves in this project's favour over time.

---

## appendix a: prior art to read before writing a line

**the direct competition, read first**
- ktransformers, "unleashing the full potential of cpu/gpu hybrid inference for moe models", sosp 2025. read the paper and the source.
- freetoken (berkeley and mit, 2026). the current state of the art on dynamic co-execution. know exactly what it does and does not claim.
- llama.cpp discussions on expert-aware ssd streaming and gpu expert cache. read the whole thread, including the objections. the objections are your requirements list.

**expert offloading and caching**
- hobbit, "a mixed precision expert offloading system for fast moe inference", 2024.
- promoe, "fast moe-based llm serving using proactive caching", 2024. the closest prior work to your prediction mechanism; you must be clearly better than it.
- pipo, "pipelined offloading for efficient inference on consumer devices", 2025.
- edgemoe, and the mixtral-offloading work, for the lru baseline you must beat.
- "two-stage expert offloading for domain-aware moe inference", ieee access 2026. the domain-correlation angle is directly relevant to your cache policy.

**foundations**
- the switch transformer and gshard papers, for routing and load balancing behaviour.
- the mixtral paper, for the expert co-activation analysis, which is the empirical basis of your disk layout.
- flexgen, for the general offloading-with-a-schedule framing.
- deepspeed-inference and zero-infinity, for the nvme-as-a-tier precedent in training.

**systems**
- the io_uring documentation and jens axboe's talks. you will live in this api.
- "what every programmer should know about memory", drepper. dated in specifics, correct in structure.
- any serious treatment of cache admission policy (tinylfu and its descendants); your cache policy is a variant of a well-studied problem and you should not reinvent it badly.

---

## appendix b: build order for one person

1. **m0, the measurement harness, on one mid-size moe model.** three weeks. produce five plots: reuse rate, access skew, oracle hit rate versus cache size, co-activation matrix, and prediction accuracy at k layers ahead.
2. **publish m0 immediately, as a standalone writeup, before building the engine.** nobody has published these numbers cleanly. it costs you nothing, it establishes you in the conversation, and it will attract the exact people you want reviewing the rest.
3. **stop here if the numbers say stop.** a clean negative result at three weeks is a good outcome. a year spent on a false premise is not.
4. m1 and m2, correctness and storage. this is unglamorous and it is where the real bandwidth comes from.
5. **m4, expert-centric prefill.** biggest win per unit of effort. do it before the fancy prediction work.
6. **m5, the speculative router head.** the actual research contribution. budget the most time here.
7. m6, m7, m8, hybrid execution, the headline benchmark, and the server.

**and one discipline throughout: publish measurements as you go, not just at the end.** in a field moving this fast, the person with the numbers gets cited even when someone else ships the better engine.
