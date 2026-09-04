# 0002. io_uring with O_DIRECT, never mmap

status: accepted, backend not yet implemented

## context

mmap plus the os page cache is what llama.cpp and most of the field use, and it
is the single most common mistake in this space.

## decision

read through an explicit async path with direct io. no mmap anywhere.

## why

- a page fault is synchronous, so the faulting thread stalls with no way to
  overlap the read against compute
- the granularity is 4kb and experts are 10 to 100mb
- the os page cache holds a second copy of bytes the cache is already holding,
  spending the ram the system is short of
- eviction becomes the kernel's decision, and the kernel has never heard of a
  router
- fault latency is invisible to the scheduler, so it cannot reason about whether
  a weight will arrive in time

the last point is the decisive one. the entire prefetch design depends on
knowing when a read will land.

## consequences

the file format is part of the product: direct io rejects unaligned offsets and
lengths outright, so 4kb alignment is a correctness constraint rather than a
tuning choice. see `strata_format::ALIGNMENT`.

the reader uses positional reads with no shared file cursor, because a shared
cursor is a false dependency between io threads and deep queues are the only way
to get real bandwidth out of consumer nvme.

**implemented and measured.** `strata-io` has the backend, and it is tested
against a portable positional-read reference for byte parity, for deep queues,
for the alignment rules, and for draining on drop. the numbers it produces are
in decision 0009.

cargo cannot link on the windows host, so the backend is built and tested inside
wsl, which is a real linux with a 6.6 kernel. that is also where `O_DIRECT` and
`io_uring` were verified to work before any of it was written.
