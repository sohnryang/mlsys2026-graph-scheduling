---
source: github
repo: owner/repo
issue_number: 86
issue_title: "Do super native tiles count towards hard fails?"
issue_url: https://github.com/yarongmu-google/MLSys/issues/86
exported_at: 2026-04-25T01:17:19Z
---

# Issue #86: Do super native tiles count towards hard fails?

## Original post
- author: Gaurav-Shah05
- created_at: 2026-04-24T18:07:26Z
- url: https://github.com/yarongmu-google/MLSys/issues/86

<comment>
@yarongmu-google Cross-ref_ [#78](https://github.com/Gaurav-Shah05/MLSys_Google/issues/78) and [#79](https://github.com/Gaurav-Shah05/MLSys_Google/issues/79). Per [#78](https://github.com/Gaurav-Shah05/MLSys_Google/issues/78) Q3, super-native granularity is unsupported in any axis. Per [#79](https://github.com/Gaurav-Shah05/MLSys_Google/issues/79), Hard Fails are limited to crash/timeout/WS-overflow. If a submission ships super-native tiles, does the evaluator (a) hard-DQ the entire submission, (b) clamp tiles to native and emit a soft warning subject to relative normalization, or (c) something else?

Thank you for taking a look at this.
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-24T18:39:42Z
- url: https://github.com/yarongmu-google/MLSys/issues/86#issuecomment-4315477506

<comment>
Thanks for the question.

Super-native granularity is a hard fail; please see Example 2 that demonstrates how to handle tensors bigger than the native granules.
</comment>

---

