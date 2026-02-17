---
source: github
repo: owner/repo
issue_number: 5
issue_title: "Clarification on the performance model"
issue_url: https://github.com/yarongmu-google/MLSys/issues/5
exported_at: 2026-02-17T08:42:35Z
---

# Issue #5: Clarification on the performance model

## Original post
- author: ericxu233
- created_at: 2026-02-05T06:59:20Z
- url: https://github.com/yarongmu-google/MLSys/issues/5

<comment>
It seems that the subgraph latency calculation is ambiguous. I have the same question as #3. The examples given aren't enough to cover every use case.

Can the organizers please clarify the exact performance model? It would be great if a formula is provided that helps us calculate subgraph_latencies and fast memory usage according to granularity and grouping, and various device statistics. It will also be helpful if an output verifier is released to us. Otherwise, it really discourages participants without any transparency on how to measure performance and correctness.

At the current state, it seems that participants are free to interpret the details of the performance as they wish...
</comment>

---

## Comment 1
- author: aheirman
- created_at: 2026-02-07T21:28:54Z
- url: https://github.com/yarongmu-google/MLSys/issues/5#issuecomment-3865512867

<comment>
I fully agree we need clarification but writing the custom verifier is kinda fun.

p.s. It sounds like a very nice contest.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T19:52:02Z
- url: https://github.com/yarongmu-google/MLSys/issues/5#issuecomment-3868102467

<comment>
Thanks for the question :)

The performance model is deterministic and fully specified in the "Objective" and "Execution Model" sections.

The Formula: As detailed in Example 1 (Strategy A), the latency for any given step is strictly the roofline bottleneck: max(Compute_Cost, Memory_Transfer_In + Memory_Transfer_Out).

The complexity of this challenge does not come from calculating the cost of a known schedule (which is simple arithmetic once the schedule is fixed), but from finding the optimal schedule (partitioning, granularities, and memory management) within the NP-hard search space.

The examples provided are not exhaustive "recipes" but demonstrations of how different scheduling choices impact that cost function. Optimizing those choices is the goal of the competition.
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-02-08T19:52:32Z
- url: https://github.com/yarongmu-google/MLSys/issues/5#issuecomment-3868103108

<comment>
I will resolve this for now. Please reopen if the above doesn't make sense.
</comment>

---

