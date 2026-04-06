---
source: github
repo: owner/repo
issue_number: 60
issue_title: "Clarification on \"only output tensors can be retained\" vs Example 3"
issue_url: https://github.com/yarongmu-google/MLSys/issues/60
exported_at: 2026-04-06T10:37:08Z
---

# Issue #60: Clarification on "only output tensors can be retained" vs Example 3

## Original post
- author: PKU-DwQ
- created_at: 2026-03-25T12:56:07Z
- url: https://github.com/yarongmu-google/MLSys/issues/60

<comment>
Hi,

I noticed a potential confusion regarding the rule:

> only output tensors can be retained

In Example 3 (Recomputation strategy), Tensor0 is used in both subgraphs:
- Subgraph 0: [0,1]
- Subgraph 1: [0,2]

However, Tensor0 is not listed in tensors_to_retain, and it is not an output of either subgraph.

From the description, it seems Tensor0 is reloaded from slow memory in each subgraph rather than retained in fast memory.

So my understanding is:
- tensors_to_retain only refers to tensors that persist in fast memory across subgraphs
- tensors not in tensors_to_retain can still be reused, but must be reloaded from slow memory

Could you confirm if this interpretation is correct?

Thanks!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T03:57:34Z
- url: https://github.com/yarongmu-google/MLSys/issues/60#issuecomment-4190220133

<comment>
Thanks for the question. Yes, your interpretation is correct. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T03:58:02Z
- url: https://github.com/yarongmu-google/MLSys/issues/60#issuecomment-4190220934

<comment>
I will resolve this for now. Please open a new issue, reference this one, if the above doesn't make sense.
</comment>

---

