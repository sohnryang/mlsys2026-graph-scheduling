---
source: github
repo: owner/repo
issue_number: 22
issue_title: "Major flaw with the problem design"
issue_url: https://github.com/yarongmu-google/MLSys/issues/22
exported_at: 2026-02-20T05:36:59Z
---

# Issue #22: Major flaw with the problem design

## Original post
- author: ericxu233
- created_at: 2026-02-13T03:24:39Z
- url: https://github.com/yarongmu-google/MLSys/issues/22

<comment>
Because the execution_granularity seems to only specify the output tensor tiling, and it seems that any arbitrary tiling can also be performed for OPs that are not output tensor OPs (Example 5 Strategy B, in the example, both tensor0 or tensor1 can be tiled), wouldn't the optimal strategy be fusing everything into one monolithic subgraph? The current problem formulation does not prevent this from happening. Or maybe I am missing some constraints here.

The problem effectively turns a scheduling problem into a tiling problem, and since we don't have a concrete way to specify any computation order or memory movement order for a large subgraph, the only job for us participants is to calculate the theoretical optimal runtime of a computational graph given the fast memory capacity constraint and utilize ephemeralism as much as possible.

Furthermore, the tensor_to_retain list seems effectively useless. For a graph like mlsys-2026-9.json, every tensor is larger than the fast memory capacity, and no tensor can be retained. This also favours fusing everything since the problem allows for ephemeral data.





</comment>

---

## Comment 1
- author: ericxu233
- created_at: 2026-02-13T03:36:20Z
- url: https://github.com/yarongmu-google/MLSys/issues/22#issuecomment-3894646920

<comment>
I may be wrong with my interpretation. Please correct me if I'm wrong.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-18T00:58:26Z
- url: https://github.com/yarongmu-google/MLSys/issues/22#issuecomment-3917877872

<comment>
Thanks for the question.

You are correct that if the problem allowed for arbitrary scheduling of individual operations, fusing the entire graph into one massive step would likely be the optimal strategy.

However, the hardware model imposes a specific constraint: the unified execution grid. As defined in the problem description, a single step (subgraph) is governed by exactly one granularity. It acts as a "master key" that enforces a rigid iteration grid on every operation within that step.

The inability to fuse everything is a direct result of the unified execution grid rule. You must break the graph into steps to accommodate different granularities, for one. There are other deeper reasons as well, but discovering them is part of the challenge.
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-02-18T00:58:48Z
- url: https://github.com/yarongmu-google/MLSys/issues/22#issuecomment-3917878902

<comment>
I will resolve this for now. Please reopen if teh above doesn't make sense.
</comment>

---

