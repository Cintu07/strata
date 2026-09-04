# 0006. the adaptive cache window is implemented and switched off

status: accepted

## context

strata trails lru on a workload of pure short-range recency, where a topic runs
long enough to turn the whole cache over and nothing is reused across topics.
caffeine's answer to exactly this is to hill-climb the size of the probationary
window against the measured hit rate, growing it when recency is paying and
shrinking it when frequency is.

## decision

implement it, measure it, and default it off.

## why

it lost ground almost everywhere it moved. from `tests/measure.rs`, hit rate
with the adaptation off and then on:

```
hot set, cap 24      0.796 -> 0.786
skewed, cap 16       0.521 -> 0.483
skewed, cap 24       0.716 -> 0.677
four domains, cap 24 0.891 -> 0.910
```

one cell improved and most got worse. a fixed 5 percent step with no decay
oscillates rather than settling, and on a small cache a growing probationary
window costs more than it can recover.

it could be tuned. tuning an adaptive controller against synthetic workloads is
fitting noise, and there are no real router traces yet.

## consequences

`CacheConfig::adaptive_window` defaults to false. the mechanism, the step
fraction and the ceiling are all still there. revisit it against real m0 traces,
which is the only thing that could justify a setting for it.
