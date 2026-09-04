# 0010. expert-major prefill must be bit identical to token-major

status: accepted

## context

expert-centric prefill turns the loops inside out: instead of walking tokens and
gathering their experts, it walks experts and applies each to every token that
wanted it. that is o3, and the measured effect is large: 68x fewer reads, 68x
fewer bytes and roughly 85x less time on the end to end benchmark.

but reordering the loops reorders the additions, and floating point addition is
not associative. the obvious implementation returns logits that differ from the
reference in the low bits.

## why that is not acceptable

g5 asks for correctness identical to a fully resident reference, verifiable by
logit diff. if the reordering itself perturbs the output, then every later diff
has a noise floor, and a real bug that moves a logit by a little is
indistinguishable from the reordering doing what it always does. the test that
was supposed to protect every subsequent optimisation stops working on the first
one.

"close enough" is also how a project ends up unable to tell whether its
quantisation, its kernels or its scheduler introduced a regression.

## decision

contributions are not accumulated as they arrive. each one is written into the
top-k slot it belongs to in a per token buffer, and the slots are summed in
index order at the end, which is exactly the order the token-major loop sums
them in. the outputs are then bit identical and the test asserts equality rather
than a tolerance.

## consequences

the cost is a contribution buffer of `n_tokens * top_k * d_model`. that is the
peak memory the prd mentions for expert-centric prefill, and
`block_size_for_budget` is what bounds it by chunking the prefill into token
blocks.

it also constrains the eventual kernels. an expert applied to a batch of rows
must produce, for each row, exactly what it would have produced for that row
alone. that is true of any row-independent ffn and of any sane gemm, but a
kernel that varies its reduction order with batch size would break the
guarantee, and this is the decision that says such a kernel is not acceptable.
