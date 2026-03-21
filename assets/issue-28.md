---
source: github
repo: owner/repo
issue_number: 28
issue_title: "Clarification on \"Unified Execution Grid\""
issue_url: https://github.com/yarongmu-google/MLSys/issues/28
exported_at: 2026-03-17T10:57:04Z
---

# Issue #28: Clarification on "Unified Execution Grid"

## Original post
- author: alex-maiorov
- created_at: 2026-02-20T09:22:46Z
- url: https://github.com/yarongmu-google/MLSys/issues/28

<comment>
Good Morning,

I apologize in advance if this is a trivial question, but I was hoping that you could produce a rigorous definition of a "Unified Execution Grid". There are numerous references to it in various github issues and the problem statement, and while I have some intuition about what this means, I have been unable to construct a rigorous model based on the information provided. If this term were defined in mathematically strict terms, a lot of my remaining confusion about the problem statement will be cleared up. 
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-03-12T01:15:18Z
- url: https://github.com/yarongmu-google/MLSys/issues/28#issuecomment-4043237373

<comment>
Thanks for the question.

Here's a more precise definition - let me know if it helps.                                                            

When you assign a granularity [w, h, k] to a subgraph, it deterministically defines how every operation in that subgraph slices its data:                                                                              

The spatial grid. The subgraph's output tensor(s) of size W × H are divided into ceil(W/w) × ceil(H/h) spatial tiles, each of size w × h. This grid is shared — all operations in the subgraph process the same spatial tile in each iteration.

Input slice shapes are derived from [w, h, k] and the operation type:
  - Pointwise inputs: w × h (same as output)
  - MatMul LHS: h × k (rows match output height, columns match reduction depth)
  - MatMul RHS: k × w (rows match reduction depth, columns match output width)
  - Output: w × h

K-steps. For MatMul operations with full reduction dimension K, the reduction is divided into ceil(K/k) steps. For Pointwise operations, k is ignored (effectively 1). If k < K, the hardware enters output-stationary mode: the output accumulator stays resident while inputs are streamed in k-sized strips.

Iteration. The system iterates over all (spatial_tile, k_step) combinations. In each iteration, every operation in the subgraph executes on the same tile. This is what makes intermediate tensors ephemeral — the output of one op at tile (i, j) feeds directly as input to the next op at the same tile (i, j), without materializing in fast memory.

The constraint. "Unified" means you cannot choose different granularities for different ops within the same subgraph. If an op can't conform to the chosen [w, h, k] (e.g., a Pointwise after a split-K subgraph where k >1), it cannot be in that subgraph.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-03-12T01:15:41Z
- url: https://github.com/yarongmu-google/MLSys/issues/28#issuecomment-4043238738

<comment>
I will resolve this for now. Please open a new issue if the above doesn't make sense.
</comment>

---

## Comment 3
- author: alex-maiorov
- created_at: 2026-03-12T12:47:16Z
- url: https://github.com/yarongmu-google/MLSys/issues/28#issuecomment-4046495891

<comment>
Thank you for your reply.

I just have one follow up question in that case. Your reply states:

`If k < K, the hardware enters output-stationary mode: the output accumulator stays resident while inputs are streamed in k-sized strips.`

Would Example 5 Strategy B not contradict this? Specifically, why is Tensor0 copied and not streamed? Otherwise, everything else makes sense. 
</comment>

---

