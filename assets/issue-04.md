---
source: github
repo: owner/repo
issue_number: 4
issue_title: "What are the possible values of op_types?"
issue_url: https://github.com/yarongmu-google/MLSys/issues/4
exported_at: 2026-02-17T08:42:28Z
---

# Issue #4: What are the possible values of op_types?

## Original post
- author: ssiu
- created_at: 2026-02-05T04:28:54Z
- url: https://github.com/yarongmu-google/MLSys/issues/4

<comment>
Hi, can we assume that `op_types` is either `MatMul` or `Pointwise`? Thanks!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T19:54:23Z
- url: https://github.com/yarongmu-google/MLSys/issues/4#issuecomment-3868105361

<comment>
Thanks for the question.

Yes, that is correct.

For this contest, op_types will strictly be either "MatMul" or "Pointwise".

You do not need to implement support for other operators like Convolution, Gather, etc.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T19:54:42Z
- url: https://github.com/yarongmu-google/MLSys/issues/4#issuecomment-3868105768

<comment>
I will resolve this for now. Please reopen if the above doesn't make sense.
</comment>

---

