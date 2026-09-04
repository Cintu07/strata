# 0008. use the io-uring crate, and allow unsafe in strata-io only

status: accepted

## context

decision 0007 says the crates carry no external dependencies and the workspace
forbids unsafe. the storage layer breaks both, so it needs its own decision
rather than a quiet exception.

## decision

`strata-io` depends on `io-uring` and `libc` on linux, and allows unsafe at the
crate level. every other crate stays dependency free and unsafe free, and the
workspace lint is `deny` rather than `forbid` so that exactly one crate can
opt out.

## why a crate and not raw syscalls

hand rolling `io_uring_setup`, the ring mmaps and the memory ordering around the
submission and completion queues is a well known source of subtle breakage, and
getting the barriers wrong produces corruption rather than a failure. the
tokio-rs `io-uring` crate is the one the rest of the ecosystem uses and it is
maintained by people who track kernel changes. writing our own would be
reinventing something badly, which is the thing the prd warns about for the
cache policy and applies just as well here.

## why unsafe is unavoidable

the entire point of the interface is to hand the kernel a buffer address and
continue doing something else. any wrapper that made that safe would have to own
the buffer for the duration, which means either a copy on completion or a
lifetime scheme equivalent to what the backend already does explicitly. a copy
of every expert would defeat the design.

the invariants are written out in the `uring` module header. the one that
matters most: **the backend drains all outstanding reads in `Drop` before the
slot pool is freed**, because otherwise the kernel writes into memory the
allocator has already reused, and that presents as corruption in an unrelated
structure minutes later.

## consequences

every unsafe block carries a `SAFETY` comment. the portable `PreadBackend` has
none, and the byte-for-byte parity test between the two backends is what would
catch the fast path going wrong.
