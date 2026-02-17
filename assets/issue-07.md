---
source: github
repo: owner/repo
issue_number: 7
issue_title: "Hardware Native Granularity for Matmul"
issue_url: https://github.com/yarongmu-google/MLSys/issues/7
exported_at: 2026-02-17T08:42:41Z
---

# Issue #7: Hardware Native Granularity for Matmul

## Original post
- author: xavierrouth
- created_at: 2026-02-05T19:06:53Z
- url: https://github.com/yarongmu-google/MLSys/issues/7

<comment>
Assuming unbounded fast memory capacity, what is the largest MatMul that can be scheduled for one iteration? There are no provided bounds on the granularity of the 'k' dimension. Assuming native granularity given as [128, 128], can the hardware execute a 128x1024x128 MatMul, with the reduction dimension in this case being k=1024 without tiling along the K dimension? The solution provided granularity would be [128, 128, 1024].
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T18:54:27Z
- url: https://github.com/yarongmu-google/MLSys/issues/7#issuecomment-3867965519

<comment>
Thanks for the question.

Yes, you are exactly right.

In this model, the native_granularity of [128, 128] strictly applies to the spatial dimensions (the output tile size). The reduction dimension (k) is flexible and is not tied to that 128 limit.

- Spatial (w, h): must match 128 to avoid padding penalties (wasteful compute).

- Reduction (k): can be any size.

- Larger k: More efficient (fewer steps), but consumes more Fast Memory.

- Smaller k: consumes less fast memory, but requires more steps.

So, [128, 128, 1024] is a perfectly valid granularity if your fast memory can hold the inputs (128x1024 LHS + 1024x128 RHS).
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T18:55:17Z
- url: https://github.com/yarongmu-google/MLSys/issues/7#issuecomment-3867968998

<comment>
I will mark this as resolved for now. Please reopen if the above doesn't make sense.
</comment>

---

