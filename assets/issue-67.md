---
source: github
repo: owner/repo
issue_number: 67
issue_title: "Validity of Split-K Execution of Fused Subgraphs"
issue_url: https://github.com/yarongmu-google/MLSys/issues/67
exported_at: 2026-04-24T11:05:36Z
---

# Issue #67: Validity of Split-K Execution of Fused Subgraphs

## Original post
- author: sohnryang
- created_at: 2026-04-06T11:17:08Z
- url: https://github.com/yarongmu-google/MLSys/issues/67

<comment>
According to clarification in #63, we can't execute subgraphs with MatMul+Pointwise fusion in split-k fashion. Then what kind of subgraphs are valid to execute in split-k tiling? I think we need some kind of formal definitions or clarifications on what kind of fusion and tiling is allowed in the problem's settings.
</comment>

---

## Comment 1
- author: sohnryang
- created_at: 2026-04-06T11:25:20Z
- url: https://github.com/yarongmu-google/MLSys/issues/67#issuecomment-4191956461

<comment>
Also, I believe that official solution validator suggested in #21 would be immensely helpful. The participants of this challenge are constantly finding many gaps in the problem statement which require additional clarifications. Instead of relying on time-consuming back-and-forth conversations via multiple GitHub issues, automatic and unambiguous validator would save everyone's time.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-07T02:06:22Z
- url: https://github.com/yarongmu-google/MLSys/issues/67#issuecomment-4196030196

<comment>
Thanks for the question.

#63 was about pointwise -> (split-k) matmul. However, the question here is about (split-k) matmul -> pointwise. These two are completely different situations. As for when split-k can be applied, please use linear algebra to decide. 

Re the validator, like I explained before, may rule out innovative scheduling. An example is the famous flash attention, which was not in anyone's "official" scheduling playbook until it's discovered.
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-04-07T02:06:43Z
- url: https://github.com/yarongmu-google/MLSys/issues/67#issuecomment-4196031228

<comment>
I will resolve this for now. Please reopen, reference this bug, if the above doesn't make sense. 
</comment>

---

