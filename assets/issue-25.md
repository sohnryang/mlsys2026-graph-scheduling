---
source: github
repo: owner/repo
issue_number: 25
issue_title: "Constraints on subgraphs"
issue_url: https://github.com/yarongmu-google/MLSys/issues/25
exported_at: 2026-02-19T01:59:46Z
---

# Issue #25: Constraints on subgraphs

## Original post
- author: sohnryang
- created_at: 2026-02-17T10:33:23Z
- url: https://github.com/yarongmu-google/MLSys/issues/25

<comment>
According to the problem statement, the hardware can execute "a sequence of connected operations" as a subgraph. I have some confusion on this definition. I can think of two possible interpretation of this statement:

1. All nodes of the subgraph must have their outputs used by only one node.
2. It is fine as long as the nodes of the subgraph are all connected and there exists some execution sequence.

The second interpretation is more relaxed when compared to the first one. For example, consider the computation graph of the example 3. Since the nodes are all connected and executing in `[op0, op1, op2]` is a valid order, the whole graph can be treated as a single subgraph when the second interpretation is used. Obviously this will allow significant improvement in latency as all tensors except `Tensor0` and `Tensor3` are ephemeral.

I personally doubt that the second interpretation is allowed because it would vastly simplify the subgraph selection, but it seems that there are no clarifications regarding this.

</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-18T01:16:59Z
- url: https://github.com/yarongmu-google/MLSys/issues/25#issuecomment-3917929293

<comment>
Thanks for the question.
    
The definition of a subgraph aligns closer to your second interpretation (connected operations), but it is strictly bounded by the unified execution grid and memory capacity rules.

You can fuse Example 3 into a single subgraph if and only if all nodes share a compatible granularity and the working set fits. If they do, then yes, you gain the latency benefit of ephemeral data.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-18T01:17:18Z
- url: https://github.com/yarongmu-google/MLSys/issues/25#issuecomment-3917930177

<comment>
I will resolve this for now. Please reopen if the above doesn't make sense.
</comment>

---

