---
source: github
repo: owner/repo
issue_number: 66
issue_title: "Clarification: MatMul LHS working set with split-K — h×K or h×k?"
issue_url: https://github.com/yarongmu-google/MLSys/issues/66
exported_at: 2026-04-07T10:37:39Z
---

# Issue #66: Clarification: MatMul LHS working set with split-K — h×K or h×k?

## Original post
- author: WalnutHo
- created_at: 2026-04-06T08:34:51Z
- url: https://github.com/yarongmu-google/MLSys/issues/66

<comment>
From #62. 

Example 5B loads Tensor0 (LHS of Op0) at full 128×128 = h×K, not in tiled form 128×32 = h×k. This isn't about the chained intermediate. Tensor0 is an external input to a non-intermediate MatMul.

For a non-chained MatMul with split-K (k < K), is the LHS working set h×K (fully resident from the first step to the end) or h×k (streamed per k-step)?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T17:22:04Z
- url: https://github.com/yarongmu-google/MLSys/issues/66#issuecomment-4193814531

<comment>
Thanks for the question.

 For a single, non-chained MatMul with split-K (k < K), the LHS working set per step is h×k, not h×K. Both inputs are streamed at their k-sliced sizes:                                                                                                                      
  - LHS: h × k per step
  - RHS: k × w per step
  - Output: h × w (accumulator, stays resident)
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T17:22:25Z
- url: https://github.com/yarongmu-google/MLSys/issues/66#issuecomment-4193816261

<comment>
I will resolve this for now. Please open a new issue, reference this one, if the above doesn't make sense.
</comment>

---

