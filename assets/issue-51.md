---
source: github
repo: owner/repo
issue_number: 51
issue_title: "Clarification: Ephemeral tensor with external consumers — is grouping valid?"
issue_url: https://github.com/yarongmu-google/MLSys/issues/51
exported_at: 2026-03-21T09:12:30Z
---

# Issue #51: Clarification: Ephemeral tensor with external consumers — is grouping valid?

## Original post
- author: tonibohnlein
- created_at: 2026-03-16T09:32:37Z
- url: https://github.com/yarongmu-google/MLSys/issues/51

<comment>
## Context

In **Example 3** (Diamond / skip-connection graph), the DAG is:
```
Op0: Tensor0 → Tensor1
Op1: Tensor1 → Tensor2
Op2: Tensor1, Tensor2 → Tensor3
```

Tensor1 is produced by Op0 and consumed by both Op1 and Op2.

## Question

Consider the subgraph decomposition `"subgraphs": [[0, 1], [2]]`.

Within subgraph `[0, 1]`, Tensor1 is an intermediate tensor (produced by Op0, consumed by Op1). Per the problem statement:

> "When operations are grouped into a subgraph, the intermediate data flowing between them becomes ephemeral, passing directly from one operation to the next without ever consuming valuable fast memory."

This makes Tensor1 ephemeral — it never materializes in fast memory.

However, Op2 (in the second subgraph `[2]`) also requires Tensor1 as input. If Tensor1 is ephemeral and was never written to fast or slow memory, how can subgraph `[2]` access it?

## Two possible interpretations

1. **Grouping is invalid.** A subgraph cannot make a tensor ephemeral if that tensor has consumers outside the subgraph. Grouping `[0, 1]` is disallowed whenever Tensor1 has external dependents. *(But the spec does not state this constraint.)*

2. **Tensor is forced to materialize.** The system detects that Tensor1 has external consumers and automatically writes it to fast memory (or slow memory), even though it is an intermediate within the same subgraph. In this case Tensor1 is not ephemeral — it occupies fast memory capacity and incurs transfer cost. *(But the spec defines all intra-subgraph intermediates as ephemeral unconditionally.)*

Which interpretation is correct?
</comment>

---

## Comment 1
- author: tonibohnlein
- created_at: 2026-03-17T11:03:49Z
- url: https://github.com/yarongmu-google/MLSys/issues/51#issuecomment-4074134115

<comment>
## Follow-up on interpretation 2: Which subgraph materializes a recomputed intermediate?

If interpretation 2 is correct (the system forces materialization when an intermediate has external consumers), consider a tensor produced by Op A with three consumers: Op B, Op C, Op D, and the schedule `[(A,B), (A,C), D]`.

In both subgraphs `(A,B)` and `(A,C)`, the tensor produced by A is an intermediate consumed within the same subgraph, so by the ephemeral rule it never touches fast memory in either case. But subgraph `D` needs this tensor as input — it must exist in fast or slow memory.

The system would need to force materialization in at least one of the two subgraphs, but there is no way to determine which one.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-03-20T03:53:30Z
- url: https://github.com/yarongmu-google/MLSys/issues/51#issuecomment-4095288158

<comment>
Thanks for the question. Interpretation 1 is correct. This is why in Example 3, we didn't group by [[0, 1], [2]].
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-03-20T03:53:56Z
- url: https://github.com/yarongmu-google/MLSys/issues/51#issuecomment-4095289799

<comment>
I will mark this as resolved for now. Please reopen an issue if the above doesn't make sense.
</comment>

---

