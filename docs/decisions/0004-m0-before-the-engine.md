# 0004. m0 comes before any engine code, and can end the project

status: accepted

## context

the design rests on empirical claims that have not been checked: that
consecutive tokens reuse experts, that routing is skewed, that experts fire in
stable groups, and above all that layer L+k routing is predictable from layer
L's hidden state.

the last one is the research contribution. if it is false, the multi-layer-ahead
prefetch cannot work and the project's central mechanism does not exist.

## decision

build and run the measurement harness first. treat its output as a go/no-go
gate, and if the numbers say stop, publish them as a short negative result and
stop.

## why

a clean negative result at three weeks is a good outcome, and nobody has
published these numbers cleanly. a year spent on a false premise is not.

## consequences

`m0/strata_m0/report.py` writes an explicit verdict table with a pass, marginal
or fail against each assumption and a headline that says plainly which way it
went. on the synthetic development trace it currently returns "stop and
reconsider", because the trained probe loses to the free persistence prior. that
is the harness working, not a bug.

the fallback if multi-layer prediction fails is written into the prd's risk
table: fall back to the persistence prior plus co-activation prefetch and
reposition around o1 and o3, which are still valuable on their own.
