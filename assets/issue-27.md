---
source: github
repo: owner/repo
issue_number: 27
issue_title: "Meaning of Base Cost / Understand Op Times"
issue_url: https://github.com/yarongmu-google/MLSys/issues/27
exported_at: 2026-03-17T10:56:44Z
---

# Issue #27: Meaning of Base Cost / Understand Op Times

## Original post
- author: als244
- created_at: 2026-02-20T02:44:44Z
- url: https://github.com/yarongmu-google/MLSys/issues/27

<comment>
In Example 5 in [problem description](https://github.com/yarongmu-google/MLSys/blob/main/PROBLEM.md) there are 2 MatMul ops that each have base cost of 2000. However in the Strategy B description: "Run Op0 and Op1. ComputeTime1 = 1,000". How is this 1000 determined? Does the base cost assume square matmuls so a 128x128 * 128x128: 2000 => 128x128 * 128x32: 500 (then * 2 because Op0 + Op1)? How would this make sense if the input native granularity is set such that W != H? Additionally if the above logic holds then it seems like there is no penalty for using smaller K and doing accumulations (128 repetitions of 128x128 * 128x1 having same throughput as 1 step of 128x128 * 128x128 does not feel practical)...?   Essentially my question is: _What does "base cost" actually refer to_?

Thanks!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-03-12T01:26:33Z
- url: https://github.com/yarongmu-google/MLSys/issues/27#issuecomment-4043274574

<comment>
Thanks for the question.

base_cost is the compute cost for executing a single operation at the native granularity with the full reduction dimension. In other words, it is the cost of one native-sized execution where the entire K dimension is consumed.

How it scales with k (reduction splitting):

The reduction dimension (k) is streamed temporally through the hardware. When you choose k < K, each step processes a proportional fraction of the dot product. The compute cost scales linearly:

compute_per_step = base_cost × (k / K)

In Example 5: both ops have base_cost = 2000, the full reduction dimension is K = 128, and the chosen granularity uses k = 32. So each step:
  - Op0 contributes: 2000 × (32 / 128) = 500
  - Op1 contributes: 2000 × (32 / 128) = 500
  - Combined: ComputeTime = 500 + 500 = 1,000

How it scales with spatial tiling (w, h):

Spatial dimensions behave differently. If you choose w or h below the native granularity, the hardware pads the execution — you pay the full base_cost per tile but produce less useful output. This increases the total number of tiles needed. In contrast, if w and h equal the native granularity (as in Example 5), there is exactly one spatial tile and no waste.

To your question about 128x128 * 128x1: there is no penalty for smaller k accumulations — the compute scales proportionally (base_cost × 1/128). This is by design: the reduction dimension streams through the array temporally, so fewer cycles = proportionally less compute. The "padding penalty" only applies to spatial dimensions.

So to summarize: base_cost is a fixed constant per operation (the cost at full native execution). Spatial tiling replicates it (with potential padding waste), while k-splitting divides it proportionally.
</comment>

---

