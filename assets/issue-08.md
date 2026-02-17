---
source: github
repo: owner/repo
issue_number: 8
issue_title: "outputs[k] is a list of tensor ids produced by Operation[k]"
issue_url: https://github.com/yarongmu-google/MLSys/issues/8
exported_at: 2026-02-17T08:42:47Z
---

# Issue #8: outputs[k] is a list of tensor ids produced by Operation[k]

## Original post
- author: aheirman
- created_at: 2026-02-07T14:53:00Z
- url: https://github.com/yarongmu-google/MLSys/issues/8

<comment>
will outputs[k] always be an array that has length 1?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T18:42:59Z
- url: https://github.com/yarongmu-google/MLSys/issues/8#issuecomment-3867903912

<comment>
Thanks for the question.

No.

While most standard operations (like standard MatMul) produce a single output, the data structure supports multiple outputs, and some benchmarks (e.g., mlsys-2026-13.json) do utilize this feature.

You should not hardcode len(outputs[k]) == 1. Your parser must handle the general case where an operation produces a list of output tensors.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T18:43:30Z
- url: https://github.com/yarongmu-google/MLSys/issues/8#issuecomment-3867906861

<comment>
I will mark this as resolved for now. Please reopen if teh above doesn't make sense.
</comment>

---

## Comment 3
- author: xavierrouth
- created_at: 2026-02-10T23:37:17Z
- url: https://github.com/yarongmu-google/MLSys/issues/8#issuecomment-3881295237

<comment>
Hi, where in the benchmarks is this feature used?
</comment>

---

