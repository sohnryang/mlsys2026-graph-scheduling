---
source: github
repo: owner/repo
issue_number: 59
issue_title: "Questions about The Implicit Reuse"
issue_url: https://github.com/yarongmu-google/MLSys/issues/59
exported_at: 2026-04-06T10:36:49Z
---

# Issue #59: Questions about The Implicit Reuse

## Original post
- author: gychen233
- created_at: 2026-03-25T10:45:51Z
- url: https://github.com/yarongmu-google/MLSys/issues/59

<comment>
I fully thank the organizers for addressing our comments, but I still have a lot of confusion.

As an extension of #15  #37 .

I would like to ask a few more questions regarding **Implicit Reuse**. 

Does 'implicit reuse' only trigger between adjacent executions, or can I **arbitrarily** retain data in the fast memory to achieve **better implicit reuse**?

Does the latency calculation assume an **optimal** memory scheduling strategy by default?

Specifically, does it use **Bélády's optimal algorithm**? 

To achieve this strategy, when freeing up space, must we evict **an entire data block** at once, or can we evict **partial slices**? 

Additionally, if a MatMul operation takes the same Tensor as both of its inputs, the fast memory needs to load **both a horizontal strip and a vertical strip** of that Tensor. Can we assume that the overlapping intersection of these strips only consumes **a single copy of capacity** within the fast memory limit?

 Finally, if our calculated `subgraph_latencies` are incorrect, will our submission receive a score of zero?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T04:15:51Z
- url: https://github.com/yarongmu-google/MLSys/issues/59#issuecomment-4190260611

<comment>
Thanks for the advanced questions. Let me address them one by one:

- Does 'implicit reuse' only trigger between adjacent executions, or can I arbitrarily retain data in the fast memory to achieve better implicit reuse? -> This problem is a (vastly) simplified abstraction from the actual TPU hardware; as such, only between adjacent. 

- Does the latency calculation assume an optimal memory scheduling strategy by default? -> Yes.

- Specifically, does it use Bélády's optimal algorithm? -> No in this simplified matter. But when you use the real TPUs, by (almost) all means, Bélády's the right scheduling algorithm. 

- To achieve this strategy, when freeing up space, must we evict an entire data block at once, or can we evict partial slices? -> entire. This is consistent with how the XLA compiler works; once you copy a slice over, it gets its own name as separate variable. (JAX is functional.)

- Additionally, if a MatMul operation takes the same Tensor as both of its inputs, the fast memory needs to load both a horizontal strip and a vertical strip of that Tensor. Can we assume that the overlapping intersection of these strips only consumes a single copy of capacity within the fast memory limit? -> sorry you can't. The overlapping intersections are duplicated copies. In real TPUs, you could copy the horizontal strip and the vertical strip as 3 separate tensors and re-assemble them back into 2 before the compute; however that will need extra ops that we "removed" in this simplified world.

- Finally, if our calculated subgraph_latencies are incorrect, will our submission receive a score of zero? -> not necessarily. The goal of this competition is to inspire people to think more about the constrained scheduling problem, not a traditional test. Therefore, all answers will be ranked against each other; deviating from the "source of truth" does not nuke an answer. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T04:16:32Z
- url: https://github.com/yarongmu-google/MLSys/issues/59#issuecomment-4190261981

<comment>
I will resolve this issue for now. Please open a new one, reference this issue, if the above doesn't make sense.
</comment>

---

