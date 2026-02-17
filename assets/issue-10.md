---
source: github
repo: owner/repo
issue_number: 10
issue_title: "compute time example 4 vs 5"
issue_url: https://github.com/yarongmu-google/MLSys/issues/10
exported_at: 2026-02-17T08:42:54Z
---

# Issue #10: compute time example 4 vs 5

## Original post
- author: aheirman
- created_at: 2026-02-07T16:37:06Z
- url: https://github.com/yarongmu-google/MLSys/issues/10

<comment>
In example 4 we pay the full base_costs per step
In example 5 we pay the proportional base_costs per step
Why?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T18:31:46Z
- url: https://github.com/yarongmu-google/MLSys/issues/10#issuecomment-3867845440

<comment>
Thanks for the question :)

This is due to the difference between spatial slicing (demonstrated in Example 4) and reduction slicing (demonstrated in Example 5). These two are essentially different linear algebra formulas.

In short: slicing the spatial dimensions (w, h) below native size incurs a padding penalty (full cost). Slicing the reduction dimension (k) does not (proportional cost).

1. Example 4: spatial penalty In this example, the chosen granularity is [64, 64, 128], but the Native Granularity is [128, 128].

You are utilizing only a quarter of the spatial compute grid (64x64 out of 128x128).

Per the execution model rules: "If you select a granularity smaller than the native size, the hardware 'pads' the execution, meaning you pay the full compute cost of the native size".

Therefore, you pay 100% of the cost for 25% of the output pixels.

2. Example 5: reduction splitting In this example, the chosen granularity is [128, 128, 32].

The spatial dimensions (128x128) match the native granularity, so the hardware is fully utilized spatially. No padding penalty applies.

You are slicing the eeduction dimension (k) (doing "Split-K"). This simply divides the mathematical workload (the dot product) into 4 temporal steps (32 out of 128).

Since the total arithmetic operations are conserved, the cost per step is proportional to the fraction of the reduction performed (32/128 = 0.25).
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T18:32:08Z
- url: https://github.com/yarongmu-google/MLSys/issues/10#issuecomment-3867847281

<comment>
I will mark this as closed for now. Please reopen if the above doesn't make sense.
</comment>

---

