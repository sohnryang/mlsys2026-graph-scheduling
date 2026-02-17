---
source: github
repo: owner/repo
issue_number: 15
issue_title: "Confusion on Reuse and Ambiguity Still exists"
issue_url: https://github.com/yarongmu-google/MLSys/issues/15
exported_at: 2026-02-17T08:43:22Z
---

# Issue #15: Confusion on Reuse and Ambiguity Still exists

## Original post
- author: ericxu233
- created_at: 2026-02-08T20:40:26Z
- url: https://github.com/yarongmu-google/MLSys/issues/15

<comment>
I fully thank the organizers for addressing our comments, but I am afraid there is still a lot of confusion and ambiguity regarding the performance model. Maybe it's just me. But I would really appreciate the organizers providing more clarification.

#5 was closed, stating that the performance model is clear with a provided formula max(Compute_Cost, Memory_Transfer_In + Memory_Transfer_Out). However, this did not address my concerns. For the contest to be fair, a given subgraph output would only have one interpretation of the performance. This provided formula neglects to mention how to compute the "Memory_Transfer_In + Memory_Transfer_Out" portion, considering reuse. There is no reuse 

The organizer's comments on #3 ambiguates this even more. The organizers mention that

> Strategy A was explicitly chosen to demonstrate a "naive reload" scenario. We wanted to show exactly what happens when you don't reuse data: you pay the full bandwidth penalty. The calculation (8,192) is correct for that specific, suboptimal choice.

>Your observation describes a more optimized strategy where you do utilize tensors_to_retain to avoid those reloads. That would indeed result in the lower latency you calculated (7,096).

They said that Example 4 Strategy A is correct, and the author's observation is also correct, but needs to utilize tensors_to_retain to avoid reload. However, the Example 4 Strategy B directly conflicts with the organizer's statement.  Example 4 Strategy A **does not utilize tensors_to_retain and doesn't assume implicit reuse within a subgraph**, but Example 4 Strategy B also **does not utilize tensors_to_retain but does assume implicit reuse**. How is this possible?

I call for the orgainzer's to clarify this. It would be nice to provide a concrete and detailed formula for subgraph_latencies on any given subgraph output based on fast_memory_capacity, slow_memory_bandwidth, native_granularity, granularities, tensors_to_retain, and traversal_orders.
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T20:50:32Z
- url: https://github.com/yarongmu-google/MLSys/issues/15#issuecomment-3868244567

<comment>
Hi @ericxu233,

Thank you for pressing on this—you are absolutely right to point out the apparent contradiction. The confusion stems from a distinction we failed to articulate clearly: Intra-Subgraph vs. Inter-Subgraph reuse.

Implicit Reuse (Intra-Subgraph):

Scope: Within a single Subgraph (a single Step in your solution).

Behavior: The hardware model automatically keeps intermediate tensors in Fast Memory (up to capacity) while executing the sequence of operations inside that step.

Example 4, Strategy B: This strategy likely "fuses" operations into a single (or fewer) Subgraphs. Because the consumer op runs in the same step as the producer, it enjoys this implicit reuse without needing any tensors_to_retain.

Explicit Reuse (Inter-Subgraph):

Scope: Across the boundary between two different Subgraphs (e.g., Step 1 -> Step 2).

Behavior: By default, Fast Memory is considered "flushed" or available for the next kernel between steps. To persist a tensor from Step 1 to be used in Step 2, you must explicitly list it in tensors_to_retain.

Example 4, Strategy A: This strategy splits the producer and consumer into separate steps. Because they are separated by a step boundary, the "naive" default is to flush and reload. To achieve reuse here, you would need tensors_to_retain.

Summary:

Strategy B relies on Fusion (grouping ops) to get implicit reuse.

Strategy A relies on Persistence (explicit management) to get reuse across boundaries.

Does this clarify the mechanics?
</comment>

---

## Comment 2
- author: ericxu233
- created_at: 2026-02-08T21:07:29Z
- url: https://github.com/yarongmu-google/MLSys/issues/15#issuecomment-3868270951

<comment>
Thanks for the response @yarongmu-google! But it is still not clear to me...maybe there is some serious confusion between us here?

Example 4 Strategy A output JSON is:

```json
{
  "subgraphs": [[0]],
  "granularities": [[64,64,128]],
  "tensors_to_retain": [[]],
  "traversal_orders": [null],
  "subgraph_latencies": [8192]
```

Example 4 Strategy B output JSON is:

```json
{
  "subgraphs": [[0]],
  "granularities": [[64,64,128]],
  "tensors_to_retain": [[]],
  "traversal_orders": [[0, 1, 3, 2]],
  "subgraph_latencies": [6548]
}
```

Based on your response, Strategy A splits produce-consumer, and Strategy B is a fused operation. What does this even mean? The example is literally only one OP. Are there intra op steps and fusion happening that the PROBLEM.md is not mentioning? Or maybe this is a misunderstanding of which example I am referring to? I am referring to "Example 4: Revisit" which the etire graph only consists of one MatMul OP.






</comment>

---

## Comment 3
- author: jerryyiransun
- created_at: 2026-02-09T02:25:50Z
- url: https://github.com/yarongmu-google/MLSys/issues/15#issuecomment-3868950595

<comment>
@yarongmu-google same question here, according to the steps in both strategies in example 4, both strategies start with the same two steps but strategy B wins because of this "implicit reuse", could you clarify what differentiates the implicit reuse?

In Strategy A:
```
Step 1 (top-left):
Move row strip 0 from the slow memory to the fast memory. MemoryTime0_in = 819.2
Move column strip 0 from the slow memory to the fast memory. MemoryTime0_in += 819.2 (MemoryTime0_in = 1638.4).
Run Op0. ComputeTime0 = Op0 = 1,500
Evict ¼ Tensor2 from the fast memory to the slow memory. MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6
TotalLatency0_1 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048
Step 2 (top-right):
Move row strip 0 from the slow memory to the fast memory. MemoryTime0_in = 819.2
Move column strip 1 from the slow memory to the fast memory. MemoryTime0_in += 819.2 (MemoryTime0_in = 1638.4).
Run Op0. ComputeTime0 = Op0 = 1,500
Evict ¼ Tensor2 from the fast memory to the slow memory. MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6
TotalLatency0_2 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048
```

In Strategy B:
```
Step 1 (top-left):
Move row strip 0 from the slow memory to the fast memory. MemoryTime0_in = 819.2
Move column strip 0 from the slow memory to the fast memory. MemoryTime0_in += 819.2 (MemoryTime0_in = 1638.4).
Run Op0. ComputeTime0 = Op0 = 1,500
Evict ¼ Tensor2 from the fast memory to the slow memory. MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6
TotalLatency0_1 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048
Step 2 (top-right):
Reuse resident row strip 0.
Move column strip 1 from the slow memory to the fast memory. MemoryTime0_in = 819.2
Run Op0. ComputeTime0 = Op0 = 1,500
Evict ¼ Tensor2 from the fast memory to the slow memory. MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6
TotalLatency0_2 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 1,500
```
</comment>

---

## Comment 4
- author: ArenaGrenade
- created_at: 2026-02-10T01:45:12Z
- url: https://github.com/yarongmu-google/MLSys/issues/15#issuecomment-3874850862

<comment>
I think this optimization only applies potentially for MatMul (Unsure how it might for Pointwise or combinations).

Stratregy 1: Traversal order is implicitly (0, 1, 2, 3)
- Step1: Load Row0 of D0 and Col0 of D1 - Memory usage: 20480
- Step2: Reuse Row0 of D0 and Load Col1 of D1 - Memory usage: 20480
- Step3: Cant reuse anything, so needs to load Row1 of D0 and Load Col0 of D1. Loading new ones means we have to evict old ones as we are already on threshold of memory usage. New memory usage: 20480
- Step4: Reuse Row1 of D0 and Load Col1 of D1

Stratregy 2: Traversal order is set to (0, 1, 3, 2)
- Step 1: Load Row0 of D0 and Col0 of D1
- Step 2: Reuse Row0 of D0 and Load Col1 of D1
- Step 3: Load Row1 of D0 and Reuse Col1 of D1
- Step 4: Reuse Row1 of D0 and Load Col0 of D1

What we can assume is the low level memory optims are handled efficiently always by the run time alg. So, just by changing what memory blocks are needed at runtime using traversal order, one can force it to use optimal memory read ops.

I dont know what the consumer producer stuff is, but the differential is in the low level memory load. As 4 blocks of traversal refers to output block calculation traversal and for each block a set of Row and Col from Data is needed. You can choose order to retain memory between traversal steps. That is why the cost differs, I think.
</comment>

---

## Comment 5
- author: yarongmu-google
- created_at: 2026-02-10T15:57:54Z
- url: https://github.com/yarongmu-google/MLSys/issues/15#issuecomment-3878848402

<comment>
Thanks for the question. Now I see where the confusion comes in.

You are both absolutely correct, and I apologize for the confusion caused by my previous "Fusion vs. Splitting" explanation.

Correction on Example 4: As @ArenaGrenade correctly analyzed, Example 4 consists of a single operation, so "fusion" (grouping multiple ops) is not the relevant factor here. The difference is indeed purely Traversal Order (Scheduling).

1. Implicit Reuse (Intra-Subgraph):

- Mechanism: This is what Strategy B utilizes. By changing the traversal_orders to [0, 1, 3, 2], the hardware processes tiles in an order where the required data happens to already be in Fast Memory from the immediately preceding tile computation.

- Rule: This reuse is automatic within the execution of a single Step. You do not need tensors_to_retain for this; you just need a smart schedule.

2. Explicit Reuse (Inter-Subgraph):

- Mechanism: This is what tensors_to_retain controls.

- Rule: If you finish a Step and want to keep data available for a subsequent Step (crossing the step boundary), you must explicitly list it in tensors_to_retain. Without this, the hardware assumes the memory is flushed/available for the next subgraph.

Summary:

- Strategy A: Less efficient because its naive traversal order [0, 1, 2, 3] constantly evicts data that is needed again shortly after within the same step.

- Strategy B: More efficient because its optimized order [0, 1, 3, 2] maximizes Intra-Subgraph (Implicit) Reuse.

We will update the PROBLEM.md to explicitly differentiate between these two scopes of reuse to align the rules with this reality. Thank you for catching this!
</comment>

---

## Comment 6
- author: yarongmu-google
- created_at: 2026-02-10T16:03:34Z
- url: https://github.com/yarongmu-google/MLSys/issues/15#issuecomment-3878902852

<comment>
I will resolve this for now. Please reopen if the above doesn't make sense.
</comment>

---

