# 0001. nvme is a tier of the memory hierarchy, not a place the model loads from

status: accepted

## context

every local inference system treats disk as where the model lives before it is
loaded. that works while the working set fits in ram. it stops working on a
16GB laptop running a 200B model, where the experts cannot all be resident at
any point and disk is touched on the critical path of every token.

## decision

treat nvme as tier 2 of the memory hierarchy, with the cache policy, the file
layout, and the scheduler all aware of it. accept that reads happen during
steady-state decode and design to hide them rather than to avoid them.

## consequences

the whole system is organised around one number, the expert cache hit rate,
because it is the divisor in `tokens/sec = bandwidth / bytes per token`.
everything else is plumbing around it.

it also fixes the competitive position. ktransformers and freetoken assume the
hot set eventually fits in ram. this is the case they do not cover, and it is
the machine most people own.
