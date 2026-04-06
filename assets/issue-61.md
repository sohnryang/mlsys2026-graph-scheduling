---
source: github
repo: owner/repo
issue_number: 61
issue_title: "Clarification on the rules of writing all output tensors back to slow memory"
issue_url: https://github.com/yarongmu-google/MLSys/issues/61
exported_at: 2026-04-06T10:37:21Z
---

# Issue #61: Clarification on the rules of writing all output tensors back to slow memory

## Original post
- author: jerryyiransun
- created_at: 2026-03-26T00:37:04Z
- url: https://github.com/yarongmu-google/MLSys/issues/61

<comment>
```
op1: T0 -> T1
op2: T2 -> T3
subgraph: [op1], [op2]
tensor to retain: [[T1], []]
```

in this example do we need to eventually write Tensor1 back to the slow memory?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T03:43:51Z
- url: https://github.com/yarongmu-google/MLSys/issues/61#issuecomment-4190194691

<comment>
Thanks for the question.

Yes, in this case, T1 is the final output so it has to go back to the slow memory.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T03:44:17Z
- url: https://github.com/yarongmu-google/MLSys/issues/61#issuecomment-4190195500

<comment>
I will mark this resolved for now. Please open a new bug, reference this one, if the above doesn't make sense. 
</comment>

---

