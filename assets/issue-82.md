---
source: github
repo: owner/repo
issue_number: 82
issue_title: "Contradicting Clarifications for #32 and #71"
issue_url: https://github.com/yarongmu-google/MLSys/issues/82
exported_at: 2026-04-23T09:45:40Z
---

# Issue #82: Contradicting Clarifications for #32 and #71

## Original post
- author: sohnryang
- created_at: 2026-04-22T16:55:43Z
- url: https://github.com/yarongmu-google/MLSys/issues/82

<comment>
The author specifically [stated](https://github.com/yarongmu-google/MLSys/issues/71#issuecomment-4293344525) that matmul -> pointwise split-k fusion is always valid. However, this directly contradicts the earlier clarification made in another issue.

> Thanks for the question.
> 
> No, this fusion is not valid with [128, 128, 32]. The entire subgraph must execute at a single granularity. With split-K, the MatMul takes 4 passes to accumulate its result, and the Pointwise cannot participate in those k-steps. 

 _Originally posted by @yarongmu-google in [#32](https://github.com/yarongmu-google/MLSys/issues/32#issuecomment-4042900526)_

Which one is correct?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-22T20:04:37Z
- url: https://github.com/yarongmu-google/MLSys/issues/82#issuecomment-4299544989

<comment>
Thanks for the question.

Revising my earlier stance — you were right to flag this as a contradiction. Short version: #71 is correct, #32 was wrong, and the confusion on my side is genuine (context below). 

What the current spec actually says.

An op belongs to exactly one subgraph, and the whole op executes inside that subgraph. So for a matmul with k < K fused with a pointwise in the same subgraph, the matmul's k-loop completes internally in that same subgraph (per spec "output-stationary mode ... locks the output slice in the fast memory as an accumulator and iterates through the input matrices in slices per the granule, paying the compute cost for each step.") The pointwise then fires once on the completed tile (per spec, "executes once per spatial tile.") Compute-neutral. Only constraint: accumulator (overlaps with O for split-k) + I/O must co-reside in fast memory.

 Same structure as Example 5 Strategy B: one subgraph, 4 k-steps inside it, one total latency. The #32 example ([128, 128, 32], K=128, matmul → pointwise) has exactly that shape and is realizable.

Why my #32 answer went the wrong way. 

An earlier version of the problem design let contestants slice an op across multiple subgraphs - in that world, a k-step could land in one subgraph and the pointwise in the next, and "pointwise can't participate in the k-steps" would have been a meaningful obstruction. That design was dropped before release in favor of the current "one op, one subgraph" format, but my #32 answer was still reasoning from the dropped model. That's on me — I should have re-derived the rule against the released spec, and I didn't.

The rule (superseding #32):
- Epilogue (matmul → pointwise): valid at any [w, h, k], provided the working set fits.
- Prologue (pointwise → matmul): requires w ≥ matmul.K (LHS) or h ≥ matmul.K (RHS), as in #71.

Apologies for the whiplash and for the loose "always" phrasing in the #71 summary that made these look inconsistent when the #71 body had already derived the right rule. A unified semantics note is coming so this gets stated once, in one place.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-22T20:05:04Z
- url: https://github.com/yarongmu-google/MLSys/issues/82#issuecomment-4299547880

<comment>
I will resolve this for now. Please open a new issue, refereeing the sone, if the above doesn't make sense.
</comment>

---

## Comment 3
- author: natetyoung
- created_at: 2026-04-23T02:56:40Z
- url: https://github.com/yarongmu-google/MLSys/issues/82#issuecomment-4301434936

<comment>
@yarongmu-google Quick check: you say "accumulator (overlaps with O for split-k) + I/O must co-reside in fast memory". For a matmul fused with a following pointwise ("pointwise epilogue"), the tensor being produced in this accumulator by the split-k matmul is ephemeral, and therefore should not contribute to the working set, correct? (Maybe this is what you mean by "overlaps with O"?)
</comment>

---

