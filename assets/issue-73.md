---
source: github
repo: owner/repo
issue_number: 73
issue_title: "Different-shaped tiles and implicit reuse"
issue_url: https://github.com/yarongmu-google/MLSys/issues/73
exported_at: 2026-04-23T09:44:26Z
---

# Issue #73: Different-shaped tiles and implicit reuse

## Original post
- author: papp-pal-andras
- created_at: 2026-04-10T21:28:55Z
- url: https://github.com/yarongmu-google/MLSys/issues/73

<comment>
Hi! Following the footsteps of #59, #65 and #70, I would like to ask for some further clarification.

These issues suggest that within a concrete step/iteration, it is allowed that two operations require a different strip of the same tensor.
Doesn't that contradict the "unified execution grid" idea, if we need to simultaneously execute the same operations on different (maybe overlapping) parts of the same tensor?
Closely related, is it indeed possible that a MatMul operation will have the same tensor as both of its inputs, as asked in #59?

Seems to me like this raises some questions. Consider a MatMul Op0 where both inputs are Tensor0, the output is Tensor1. Both tensors are 512x512, and we use execution granularity [128, 128, 512] (no split-k). In each step, we always need one row-strip and one column-strip from Tensor0. How would e.g. implicit reuse be defined in this case? Would the row strip be compared to the last row strip, and the column strip to the last column strip, as if they were independent tensors?

In a more complex case, we could have column strips of different widths from the same tensor, e.g.
Tensor0, Tensor1 -> Op0 (MatMul) -> Tensor3
Tensor2, Tensor1 -> Op1 (MatMul) -> Tensor4
Tensor3, Tensor4 -> Op2 (MatMul) -> Tensor5
All tensors 512x512, execution granularity [128, 128, 256]. Op0 uses column strips of width 256 from Tensor1 due to split-k, Op1 uses column strips of width 128 from Tensor1 due to spatial tiling. How would implicit reuse work here, is the last tile remembered per shape?

It seems to me that allowing different tiles of the same matrix in the same step may create some ambiguity for this simplified model.

Thanks a lot in advance!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-22T22:00:57Z
- url: https://github.com/yarongmu-google/MLSys/issues/73#issuecomment-4300232484

<comment>
Thanks for the question.

The model is simpler than the question suggests: all inputs needed by the ops in a subgraph must be resident in fast memory simultaneously, and the working set is their sum. Identical uses of the same slice dedup (count once); anything that differs in tensor, role, extent, or indices is a separate slice and counted separately. So if you have the full tensor in the fast memory, you can use any slice of it. But if you have one strip that's not overlapping with a diff strip, you have to reload. This is how common compilers handle memory.

Applied to your Example 1 (T0 @ T0, 512², granule [128, 128, 512]): at each iteration the matmul needs an LHS row-band (T0, rows=r·128..(r+1)·128, all cols) and an RHS col-band (T0, all rows, cols=c·128..(c+1)·128). Same tensor, different slices — two entries, working set includes both concurrently. They'd only dedup if they were the exact same slice, which they're not.                      

Applied to Example 2 (Op0 reads T1 as width-256 RHS; Op1 as width-128 RHS): different slices → two entries, both resident, both priced in the subgraph's working set. No cross-reuse between them.
                                                                                                                       
Practical note: none of our 24 benchmarks (released + held-out) contain any op with duplicate inputs, so self-matmul and the multi-role-single-tensor stress case are hypothetical - the model resolves them cleanly but they don't appear in grading.

</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-22T22:01:16Z
- url: https://github.com/yarongmu-google/MLSys/issues/73#issuecomment-4300233895

<comment>
I will resolve this for now. Please open a new issue, citing this one, if the above makes sense.
</comment>

---

