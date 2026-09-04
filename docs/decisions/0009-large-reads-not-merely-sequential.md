# 0009. what the layout is optimising for is large reads, not adjacency

status: accepted, arrived at by measurement

## context

the prd's storage argument is that random access is catastrophic and sequential
access is survivable, and the co-activation layout exists to convert one into
the other. that framing is right in direction and wrong in detail, and the
measurement says so.

## the measurement

from `cargo run --release -p strata-io --bin bandwidth`, on this machine's nvme
through a wsl2 ext4 volume:

```
pattern      block   qd      GB/s        IOPS     lat us
sequential      1M    1     1.737        1656      603.8
sequential      1M   16     3.514        3352     4774.0
random          4K    1     0.023        5495      182.0
random          4K  128     0.406       99181     1290.6
random         64K   16     2.496       38089      420.1
random          1M    4     3.843        3665     1091.5
random          1M   64     3.835        3657    17501.0
```

three things fall out of this.

**queue depth is worth 18x on small random reads.** at queue depth one the
device is idle most of the time waiting for a round trip. this is why the
storage interface is submit-then-wait and not a blocking `read_at`: a blocking
interface cannot express depth greater than one, and adopting one anywhere in
the engine would cost most of the device.

**random 1M is as fast as sequential 1M.** 3.84 GB/s against 3.51. once a
request is large enough, where it sits stops mattering. the penalty is a fixed
cost per request, not a cost of moving the head.

**the crossover is somewhere between 64k and 1M.** random 64k reaches 2.5 GB/s
at depth 16 and random 1M reaches 3.8 GB/s at depth 4.

## decision

state the goal as **making reads large**, not as making them contiguous.
adjacency is the means, not the end.

## consequences

this changes what the co-activation layout is for. it is not there to make the
head sweep forwards; the device has no head. it is there so that experts wanted
together fall inside one large request, which reduces the request count.

it also predicts, correctly, where coalescing stops paying. the end to end
benchmark uses 2 MiB experts and shows the coalesced stage merging 482 reads
into 43 with no time saved at all, because at 2 MiB each read was already past
the point where size stops mattering. coalescing earns its place when experts
are small, which is the fine-grained-expert model families and the low bit
quantisations, and those are exactly the cases the project cares about most.

the caveat on the absolute numbers: this ran inside wsl2, so an ext4 volume on a
virtual disk on ntfs, with both layers in the path. the shape of the curve is
real and the peak is a lower bound on the bare device.
