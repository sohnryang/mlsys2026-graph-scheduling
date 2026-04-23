---
source: github
repo: owner/repo
issue_number: 78
issue_title: "Follow-up Questions about Native_granularity"
issue_url: https://github.com/yarongmu-google/MLSys/issues/78
exported_at: 2026-04-23T09:45:14Z
---

# Issue #78: Follow-up Questions about Native_granularity

## Original post
- author: gychen233
- created_at: 2026-04-21T03:02:55Z
- url: https://github.com/yarongmu-google/MLSys/issues/78

<comment>
A follow-up questions of #74 .

**Q0**: Should the cost for #74 be calculated as $2000 \cdot \frac{256}{128} + 2000 \cdot \frac{32}{128} = 4500$?"

I would like to inquire about another example to further clarify the definition of native_granularity.

<img width="1798" height="819" alt="Image" src="https://github.com/user-attachments/assets/1c3f5fa8-a8f7-4c9c-9b87-cf342dafca1d" />

All operators in the example are matrix multiplications.

**Q1**: Assuming the native_granularity is $[128, 128]$, is it possible to fuse all operators and set the fused granularity to $[128, 128, 128]$?

**Q2**: If so, would the computation time for a single iteration be calculated as follows?

$2000 \cdot \frac{256}{128} \cdot \frac{256}{128} + 2000  \cdot \frac{256}{128} + 2000 = 14000$

**Q3**: Is it permissible for granularity[2] to be strictly greater than native_granularity[2]?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-22T21:02:28Z
- url: https://github.com/yarongmu-google/MLSys/issues/78#issuecomment-4299949398

<comment>
Thanks for the question.

Re Q0: No.

Re Q1: yes, since you are setting the chosen granule to be the same as the native granule.

Re Q2: not sure what a "single iteration" means so I don't know how to answer your question. If all mammals are fused into one subgraph, in one possibility, your [256, 256]@[256, 128] takes 8 steps to compute, [128, 256]@[256, 128] takes 4 steps to compute - can you clarify which one is our "single iteration"?

Re Q3: no. k ≤ native_k in the same way w ≤ native_w and h ≤ native_h. PROBLEM.md describes the native granule as "the hardware's native execution granularity" that applies uniformly across all three dimensions (line 33), and every worked example schedules at native or below. We don't support super-native in any of the three axes. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-22T21:02:47Z
- url: https://github.com/yarongmu-google/MLSys/issues/78#issuecomment-4299951092

<comment>
I will resolve this for now. Please open a new issue, citing this one, if the above doesn't make sense. 
</comment>

---

