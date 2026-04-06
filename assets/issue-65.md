---
source: github
repo: owner/repo
issue_number: 65
issue_title: "Question about intra-subgraph tensor reuse"
issue_url: https://github.com/yarongmu-google/MLSys/issues/65
exported_at: 2026-04-07T10:36:44Z
---

# Issue #65: Question about intra-subgraph tensor reuse

## Original post
- author: Richard1688Sun
- created_at: 2026-04-06T08:09:13Z
- url: https://github.com/yarongmu-google/MLSys/issues/65

<comment>
Extending off #59 

In that issue it was mentioned that if a MatMul uses the same tensor for both inputs, we **cannot** assume the overlapped intersection consume a single copy of capacity.

Is this the same if 2 fused operations uses the same full input tensor? In this case, can we assume that the common input tensor only consumes a single copy? Or we need to count both separately?

eg. We fuse Op1 and Op2 into 1 subgraph
```
T1 -> Op1 -> T2
T1 + T3 -> Op2 -> T3
```
> Note: Where Op1 is pointwise and Op2 is matmul

Would the scenario change if this subgraph required tiling vs if it didn't?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T16:19:29Z
- url: https://github.com/yarongmu-google/MLSys/issues/65#issuecomment-4193431557

<comment>
Thanks for the question. In your example, T1 can be reused. #59 was asking about partial reuse of a tensor moved across memory space, while here is full tensor reuse.

In general, when a compiler moves a tensor from one memory space to another, it's often renamed; retained tensors with the same (re)name can be reused. However, if a compiler moves a partial tensor, that partial tensor gets its own new name - thus partial reuse is not feasible.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T16:19:52Z
- url: https://github.com/yarongmu-google/MLSys/issues/65#issuecomment-4193433986

<comment>
I will resolve this for now. Please open a new issue, reference this one, if the above doesn't make sense.
</comment>

---

