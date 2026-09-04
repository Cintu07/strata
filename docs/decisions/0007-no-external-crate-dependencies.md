# 0007. the rust crates have no external dependencies

status: accepted

## context

serde and bincode would have written the layout file's index in a few lines. a
crc crate, a hashmap crate and a rand crate would each have saved a little code.

## decision

zero external dependencies in `strata-format`, `strata-cache` and
`strata-layout`. std only.

## why

- **the file is an on-disk contract.** other processes, and eventually other
  languages, read it. the byte layout should be visible in the source and
  reviewable as a specification, not implied by a derive macro whose encoding
  can shift under a version bump.
- **the workspace builds and tests offline.** for a systems project that will be
  built on constrained and disconnected machines, that is worth something.
- **the code in question is small.** crc32 is thirty lines, the little-endian
  codec is sixty, and both are fully tested against known vectors.

## consequences

test helpers write their own deterministic xorshift generators rather than
pulling in `rand`, which has the side benefit that every failing test is exactly
reproducible.

this rule is not permanent and not a virtue in itself. `tokio` or `io-uring` for
m2 would be entirely reasonable. the rule is that adding one requires a decision
record saying why.
