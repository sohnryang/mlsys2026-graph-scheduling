---
source: github
repo: owner/repo
issue_number: 62
issue_title: "Clarification: MatMul LHS memory behavior during split-K"
issue_url: https://github.com/yarongmu-google/MLSys/issues/62
exported_at: 2026-04-06T10:37:32Z
---

# Issue #62: Clarification: MatMul LHS memory behavior during split-K

## Original post
- author: WalnutHo
- created_at: 2026-03-27T06:03:09Z
- url: https://github.com/yarongmu-google/MLSys/issues/62

<comment>
The spec in problem.md defines the LHS input slice shape as:

> the Left-Hand Side (LHS) input requires width **k** (reduction depth) and height **h**

This suggests the working set contribution of LHS is `h × k` per tile.

However, in Example 5 (Split-K with `k=32`, `K=128`):

> We keep **Tensor0 (128×128)** and the accumulator Tensor4 (128×128) resident. We stream Tensor1 (128×32 strip) and Tensor2 (128×32 strip). Total Working Set is **40,960**.

Verification:
- If LHS = h×K: `128×128 + 128×128 + 128×32 + 128×32 = 40,960` ✓
- If LHS = h×k: `128×32 + 128×128 + 128×32 + 128×32 = 28,672` ✗

The example appears to use `h × K` (full reduction dimension), not `h × k`.

Since the output format only exposes `[w, h, k]` and `traversal_orders` — with no field to control per-tensor loading strategy — this must be a fixed hardware behavior rather than a solver choice.

Could you clarify:

1. Is it always the case that LHS is loaded at full size `h × K` (resident across k-steps), while RHS is streamed at `k × w` per step? Or does the hardware stream both inputs at their k-sliced sizes (`h × k` and `k × w`)?

2. If LHS is always fully loaded, should Example 4's description of LHS as "row strip 0 (64×128)" also be interpreted as the full `h × K` slice (which happens to equal `64 × 128` since `k = K = 128` in that example)?

Thank you!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T03:42:14Z
- url: https://github.com/yarongmu-google/MLSys/issues/62#issuecomment-4190191926

<comment>
Thanks for the question.

Example 5 shows a chained matmul, and split-k is applied in the 2nd matmul. The LHS of the second matmul is in fact the result of the first matmul: h x k.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T03:42:42Z
- url: https://github.com/yarongmu-google/MLSys/issues/62#issuecomment-4190192717

<comment>
I will resolve this for now. Please open a new bug, reference this one, if the above doesn't make sense.
</comment>

---

