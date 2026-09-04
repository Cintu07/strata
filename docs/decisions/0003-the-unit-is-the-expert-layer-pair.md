# 0003. the unit of everything is the expert-layer pair

status: accepted

## context

it is tempting to talk about "expert 5" and to index caches, traces and layouts
by expert number.

## decision

`ExpertKey { layer, expert }` everywhere. there is no api that lets you name an
expert without naming its layer.

## why

expert 5 in layer 3 and expert 5 in layer 30 are unrelated tensors that happen
to share an index. merging them makes every statistic downstream meaningless:
reuse rates get inflated by collisions between layers, co-activation matrices
mix graphs that have nothing to do with each other, and cache hit rates count
hits that never happened.

this is cheap to get right at the start and expensive to discover later, because
the failure is silently plausible numbers rather than a crash.

## consequences

the type is in `strata-format` and both other crates and the python harness use
it. `ExpertKey::packed` gives the dense u64 that traces and sketches want, and
`m0`'s `RouterTrace.flat_keys` packs identically, so a trace written in python
replays through the rust cache without a translation step that could quietly
disagree.
