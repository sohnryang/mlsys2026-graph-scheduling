---
source: github
repo: owner/repo
issue_number: 38
issue_title: "Scaling of computation costs in Example 2"
issue_url: https://github.com/yarongmu-google/MLSys/issues/38
exported_at: 2026-03-17T11:00:03Z
---

# Issue #38: Scaling of computation costs in Example 2

## Original post
- author: papp-pal-andras
- created_at: 2026-03-11T16:12:41Z
- url: https://github.com/yarongmu-google/MLSys/issues/38

<comment>
Hi! Issue #10 clarifies the scaling of the base_costs for Examples 4 and 5. Specifically, in Example 5, the base costs scale down proportionally with reduction splitting.

The same scaling remains unclear to me in Example 2. The descriptions suggest that the ComputeTime of Op0 remains 1000 in all the 4 steps in Strategies A and B. However, the chosen granularities are neot below the native granularity, so it is unclear to me why the compute costs do not scale down proportionally (to 250 in each step in 2A, 275 in each step in 2B), similarly to Example 5. Is that maybe a mistake in the solution explanation?

Thanks for the help!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-03-11T19:49:28Z
- url: https://github.com/yarongmu-google/MLSys/issues/38#issuecomment-4041776841

<comment>
Thanks for the question! This is not an error - the difference comes from the nature of the two dimensions  being split.                                                                                                
                                                                                                              
Example 2 (Pointwise, spatial tiling): The tensors are 256x256 and the native array is 128x128. Each 128x128 tile is a full spatial pass through the physical array, so each tile pays the full base_cost. 4 tiles × 1,000 = 4,000 total compute. This is the expected cost — no waste, just more passes to cover the larger tensor.

Example 5 (MatMul, k-splitting): Here the spatial granularity matches native (128x128), but the reduction dimension K=128 is split into steps of k=32. The k dimension is streamed temporally through the array - fewer elements, fewer cycles - so each step costs base_cost × (k/K). 4 steps × 500 = 2,000 total compute per op. Same total work, just spread over more steps.

The key distinction is spatial vs. temporal: spatial tiling at or above native doesn't reduce per-tile compute (the full array fires each time), while k-splitting divides compute proportionally (fewer cycles streamed). 

Given the wider than expected confusion caused by the spatial vs temporal distinction, we've just updated PROBLEM.md to call this out explicitly; see the paragraph starting "note that the hardware has a native execution granularity. We've also updated the benchmark datasets to lower the difficulty by enforcing that the native_granule is the same across all dimensions. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-03-11T19:49:49Z
- url: https://github.com/yarongmu-google/MLSys/issues/38#issuecomment-4041779106

<comment>
I will resolve this for now. Please follow up if the above doesn't make sense.
</comment>

---

