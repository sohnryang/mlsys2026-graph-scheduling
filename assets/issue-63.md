---
source: github
repo: owner/repo
issue_number: 63
issue_title: "Subgraph execution clarifications"
issue_url: https://github.com/yarongmu-google/MLSys/issues/63
exported_at: 2026-04-06T10:37:46Z
---

# Issue #63: Subgraph execution clarifications

## Original post
- author: natetyoung
- created_at: 2026-04-01T00:00:42Z
- url: https://github.com/yarongmu-google/MLSys/issues/63

<comment>
I have two clarification questions about subgraphs, which stem from how I'm imagining they might be executing physically in a real system. There are some constraints and costs which seem like they would come up, but are not stated explicitly in PROBLEM.md.

1. Is there a memory limit for ephemeral tensor slices? Consider example 3: if all 3 operations are fused into the same subgraph, are both tensor1 and tensor2 ephemeral simultaneously (which would seem to correspond to having multiple matrix registers in the compute core), or does tensor1 have to be placed in the fast memory while tensor2 is being produced? Can you ever run out of space for simultaneously-held slices of ephemeral tensors?

2. What happens if we try to fuse a pointwise into a following matmul in a way which would require input reloading? Consider, for instance, a 256x256 pointwise which happens before a 256x256x256 matmul, with a native execution granularity of 128x128. If these are fused into the same subgraph, the largest possible execution granularity for the subgraph is 128x128x128 (since a larger reduction dimension would create a larger-than-native-granularity tile for the pointwise operation). However, at this granularity, the matmul has to re-read the input matrices as it executes (since it will have to evict both its A and B slices to move onto the next while accumulating the output). What does this mean for the pointwise operation? I can see 4 possibilities:

- The pointwise will have to be recomputed on a tile whenever the matmul needs the tile again after forgetting it, incurring extra compute cost.
- The output of the pointwise must be stored in the fast memory (at whatever granularity would be necessary to avoid recomputation).
- These operations simply cannot be part of the same subgraph. This situation is already forbidden by the "compatible granularity" constraint.
- This is a simplified problem anyway and we are ignoring this and pretending that somehow the outputs to the pointwise operation are remembered.

Also, it could be the case that I simply have some kind of misunderstanding here. In any event, I am also in favor of an official solution validator as suggested in #21, which would of course clear both of these up.
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T03:15:57Z
- url: https://github.com/yarongmu-google/MLSys/issues/63#issuecomment-4190141426

<comment>
Thanks for the questions.

Re #1: This problem is a much simpler abstraction from the real TPU hardware, so as PROBLEM.md states: "Ephemeral data exists when operations are grouped into a single subgraph.". Note that ephemerality is just a concept, not a real memory in the problem, and therefore has no size limit etc.  The abstracted hardware only has the fast memory and the slow memory as concrete, physical "registers".

Re #2: This is your option 3. MatMul+Pointwise fusion is not valid when split-K is required (k < K). The entire subgraph executes at a single granularity, and the Pointwise cannot participate in the MatMul's k-step accumulation loop.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T03:16:25Z
- url: https://github.com/yarongmu-google/MLSys/issues/63#issuecomment-4190142522

<comment>
I will mark this as resolved for now. Please open another issue if the above doesn't make sense.
</comment>

---

