---
source: github
repo: owner/repo
issue_number: 69
issue_title: "Can Ephemeral Tensors Be Retained for the Next Subgraph?"
issue_url: https://github.com/yarongmu-google/MLSys/issues/69
exported_at: 2026-04-07T10:38:38Z
---

# Issue #69: Can Ephemeral Tensors Be Retained for the Next Subgraph?

## Original post
- author: Richard1688Sun
- created_at: 2026-04-06T23:50:23Z
- url: https://github.com/yarongmu-google/MLSys/issues/69

<comment>
If a tensor is ephemeral for subgraph0, can it be retained to subgraph1? In such case, is the data no longer ephemeral since it takes up fast memory?

#64 noted that ephemeral tensors are store temporarily in registers, can these be later written into fast memory and retained for the next subgraph?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-07T02:10:09Z
- url: https://github.com/yarongmu-google/MLSys/issues/69#issuecomment-4196042547

<comment>
Thanks for the question.

Yes, you can choose to retain an ephemeral to be reused in the n=subsequent subgraph. In that case, indeed it will take up fast memory. 

#65 included some brief info on how a real hardware works, in which register values could be evicted into the fast memory. However, please note that registers are a concept that do not exist in this abstraction. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-07T02:10:36Z
- url: https://github.com/yarongmu-google/MLSys/issues/69#issuecomment-4196044075

<comment>
I will resolve this for now. Please open a new bug, reference this one, if the above doesn't make sense.
</comment>

---

