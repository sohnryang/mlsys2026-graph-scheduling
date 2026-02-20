---
source: github
repo: owner/repo
issue_number: 20
issue_title: "Multiple Outputs, Granularity and Iteration Order"
issue_url: https://github.com/yarongmu-google/MLSys/issues/20
exported_at: 2026-02-20T05:36:45Z
---

# Issue #20: Multiple Outputs, Granularity and Iteration Order

## Original post
- author: xavierrouth
- created_at: 2026-02-11T22:59:03Z
- url: https://github.com/yarongmu-google/MLSys/issues/20

<comment>
Hi, I was wondering how granularity and iteration order apply to subgraphs with multiple tensor outputs. It seems like this can cause contradictions in the inferred granularities of other operations in the subgraph, and I'm not sure if this is valid / how to handle these. 

If we have two output tensors, A (256 x 256) and B (128 x 256), then do we specify an iteration order of 4 elements [0, 1, 2, 3] or of 2 elements [0, 1]? Can we specify the iteration orders per output tensor of the subgraph? Similarly, can we specify the granularity per output tensor / operation of the subgraph? What happens if these granularities conflict with each other, or if the number of iterations implied by the granularities conflicts? 
 
In general, it is not clear from the examples how the granulartiy affects subgraphs with multiple output operations. 
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-18T00:39:13Z
- url: https://github.com/yarongmu-google/MLSys/issues/20#issuecomment-3917826907

<comment>
Thanks for the question.

The granularity creates a single unified execution grid for the entire subgraph. Every operation within that step must conform to this grid. If you find yourself with outputs that require different granularities, we recommend re-reading the Objective section in PROBLEM.md; for example, "Your task is to... that covers every operation in the graph at least once ..."


</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-18T00:39:29Z
- url: https://github.com/yarongmu-google/MLSys/issues/20#issuecomment-3917827658

<comment>
I will resolve this for now. Please reopen if the above doesn't make sense.
</comment>

---

## Comment 3
- author: xavierrouth
- created_at: 2026-02-18T14:21:18Z
- url: https://github.com/yarongmu-google/MLSys/issues/20#issuecomment-3921111999

<comment>
Does this mean 
1) all subgraph outputs must be the same size? or 
2) all subgraph outputs must be covered by the same number of iterations. For example if you had granularity 128x16x1, and you had two tensor outputs of 128x128 and 64x128, the unified iterations would be 8, but the tensor outputs would be different sizes. Is this valid?

We can't reopen issues.
</comment>

---

## Comment 4
- author: xavierrouth
- created_at: 2026-02-19T22:33:27Z
- url: https://github.com/yarongmu-google/MLSys/issues/20#issuecomment-3930529264

<comment>
Hi, additionally, if a tensor is an output of a subgraph, but is also used ephemerally inside the subgraph, how does this affect the execution granularity of the subgraph? Can this output cause conflicts with other outputs due to granularity?

</comment>

---

