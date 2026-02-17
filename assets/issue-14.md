---
source: github
repo: owner/repo
issue_number: 14
issue_title: "Can we combine MatMul and Pointwise in a single subgraph?"
issue_url: https://github.com/yarongmu-google/MLSys/issues/14
exported_at: 2026-02-17T08:43:15Z
---

# Issue #14: Can we combine MatMul and Pointwise in a single subgraph?

## Original post
- author: aheirman
- created_at: 2026-02-08T13:43:26Z
- url: https://github.com/yarongmu-google/MLSys/issues/14

<comment>
The current rules do not discuss if this is permitted.
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T17:55:34Z
- url: https://github.com/yarongmu-google/MLSys/issues/14#issuecomment-3867752392

<comment>
Yes.

This is permitted as long as the total working set for the fused subgraph fits within fast_memory_capacity.

Note that the entire subgraph will execute with a single granularity [w, h, k]. It is your responsibility to ensure that fusing a Pointwise op into a potentially tiled reduction (Split-K) remains mathematically valid.
</comment>

---

