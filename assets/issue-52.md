---
source: github
repo: owner/repo
issue_number: 52
issue_title: "Clarification: Can graph input tensors be retained across subgraphs?"
issue_url: https://github.com/yarongmu-google/MLSys/issues/52
exported_at: 2026-03-21T09:12:37Z
---

# Issue #52: Clarification: Can graph input tensors be retained across subgraphs?

## Original post
- author: tonibohnlein
- created_at: 2026-03-16T11:02:09Z
- url: https://github.com/yarongmu-google/MLSys/issues/52

<comment>
## Context

In a [previous issue response](https://github.com/yarongmu-google/MLSys/issues/34), the organizer clarified that retained tensor lifetime extends exactly one step:

> "To keep a tensor alive for step k+2, step k+1 would need to **also produce it** and explicitly retain it in `tensors_to_retain[k+1]`."

The key phrase is "also produce it." Graph input tensors are never produced by any operation — they simply exist in slow memory at the start of execution.


## Question

Consider benchmark 13, where Tensor 0 (4096×4096) is a graph input shared across 16 parallel MatMul chains.

If I schedule these chains as separate subgraphs, each one loads and consumes Tensor 0 but no op ever *produces* it. Under the stated rule, does this mean graph inputs can never appear in `tensors_to_retain`? If so, the only way to avoid reloading a shared graph input for every consumer subgraph would be to group all consumers into a single subgraph.

Is this the intended behavior? Or can a tensor that is already resident in fast memory (via a previous `tensors_to_retain`) be re-retained without being produced by the current subgraph?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-03-20T03:42:58Z
- url: https://github.com/yarongmu-google/MLSys/issues/52#issuecomment-4095247568

<comment>
Thanks for the question.

For benchmark #13, yes, if you want to avoid reloading the shared input, you will need to group all consumers into a single subgraph. 

Sorry for the confusion. The root cause was that originally there was a way to retain pure inputs in the fast memory through a different mechanism. But our review says that makes the problem too complicated, so we removed that mechanism. My sincere apologies!
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-03-20T03:43:21Z
- url: https://github.com/yarongmu-google/MLSys/issues/52#issuecomment-4095248796

<comment>
I will resolve this for now. Please open a new bug if the above doesn't make any sense.
</comment>

---

