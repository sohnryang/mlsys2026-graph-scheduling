---
source: github
repo: owner/repo
issue_number: 33
issue_title: "Clarification in Example 2"
issue_url: https://github.com/yarongmu-google/MLSys/issues/33
exported_at: 2026-03-17T10:58:16Z
---

# Issue #33: Clarification in Example 2

## Original post
- author: adassarma
- created_at: 2026-02-25T04:15:15Z
- url: https://github.com/yarongmu-google/MLSys/issues/33

<comment>
In example 2, both the strategies seem to directly contradict the requirement "For any chosen execution granularity, the sum of the required input slices and the resulting output slices must fit simultaneously within the fast memory capacity.", since 16384(for input tensor)+16384(for produced output tensor) exceeds fast memory capacity of 25000. Am I thinking about this correctly or does this mean that for pointwise ops this requirement is relaxed?
</comment>

---

## Comment 1
- author: papp-pal-andras
- created_at: 2026-03-11T15:26:37Z
- url: https://github.com/yarongmu-google/MLSys/issues/33#issuecomment-4040024707

<comment>
I was also wondering about this.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-03-11T23:21:10Z
- url: https://github.com/yarongmu-google/MLSys/issues/33#issuecomment-4042853582

<comment>
Thanks for the question.

You're absolutely right — this is a bug. With fast_memory_capacity: 25000 and granularity [128, 128, 1], the working set for a single Pointwise op is 2 × 128×128 = 32,768, which exceeds the capacity. This should have been caught before publishing.
                                                                                                              
The root cause: an earlier version of the problem included memory donation, but we removed that to simplify the problem and forgot to update this example's capacity accordingly.                                                                             

This is now fixed - capacity has been increased to 35,000 and the example has been renamed to "Larger Tensors" since the focus is on multi-step execution with bigger tensors, not memory pressure. The math for all strategies is unchanged.                                                                                

Thanks for catching this, and apologies for the confusion.
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-03-11T23:21:32Z
- url: https://github.com/yarongmu-google/MLSys/issues/33#issuecomment-4042854824

<comment>
I will resolve this for now. Please open another issue if the above doesn't make sense.
</comment>

---

