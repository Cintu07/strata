# 0005. the cache policy is w-tinylfu admission over a gdsf main region

status: accepted, arrived at by measurement

## context

the prd asks for cost-aware frequency with recency decay, admission on second
touch, and segmented residency. the first implementation took that literally:
a lifetime access count per resident entry, an eviction score of
`clock + freq * reload_cost / bytes`, and admission gated on whether a newcomer
outscored the entry it would evict.

it deadlocked. after one topic has run for a while its experts carry counts in
the tens and every newcomer arrives with one, so nothing is admitted, so nothing
is evicted, so the aging clock never advances, and the cache freezes on whatever
it saw first. on a four-domain workload it scored 0.25 against lru's 0.96.

## decision

- eviction stays greedy dual size frequency. the clock that rises to whatever
  was last evicted is the recency decay, and it needs no sweep over the table.
- the reload cost **must include the fixed request latency**. modelled as pure
  bandwidth, `cost / size` is the constant `1 / bandwidth` and the entire term
  cancels, leaving a popularity contest. with latency in it, a small expert is
  correctly worth more per byte than a large one.
- frequency comes from a 4-bit count-min sketch with periodic halving, fed by
  **every access including misses**, not from a lifetime counter. an expert
  returning after a digression has recent history and none of it is resident.
- admission compares sketch estimates on **both** sides, so it is measuring the
  same window for the candidate and the victim.
- new experts enter a small probationary window and only contend when they fall
  out of it, so a burst is absorbed rather than fought.

that combination is w-tinylfu with a gdsf main region. it is a known algorithm
with a known name, deliberately. the prd says not to reinvent this badly.

## the sizing bug worth recording

the sketch's counter table is sized by the model's expert universe, so unrelated
experts rarely share a counter. its decay window is sized by **how many experts
the cache holds**, so a topic that fell out of use fades before the cache turns
over. sizing the window by the universe was a second version of the same
deadlock: with 4096 experts and a cache holding 12, the counters never halved
inside a whole run.

since expert sizes are not known until they arrive, the cache tracks a running
mean of admitted sizes and retunes the window from it.

## consequences

on synthetic workloads: clearly ahead of lru on one-shot scan traffic and on
skewed access, behind lru on pure short-range recency. that last case is
asserted as a test rather than omitted, and the `tinylfu_admission` flag is the
ablation that shows the cost is admission specifically.
