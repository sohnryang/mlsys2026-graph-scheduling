---
source: github
repo: owner/repo
issue_number: 74
issue_title: "Questions about native_granularity and granularities"
issue_url: https://github.com/yarongmu-google/MLSys/issues/74
exported_at: 2026-04-23T09:44:32Z
---

# Issue #74: Questions about native_granularity and granularities

## Original post
- author: gychen233
- created_at: 2026-04-13T08:48:43Z
- url: https://github.com/yarongmu-google/MLSys/issues/74

<comment>
I'm a little confused by the concept of granularity. I appreciate you taking the time to answer these few questions for me.

**Q1:** Can **granularity[0]** be greater than **native_granularity[0]** ? ( And can **granularity[1]** be greater than **native_granularity[1]** ? )

I have an example here that I'd like you to take a look at.

<img width="1864" height="1085" alt="Image" src="https://github.com/user-attachments/assets/a77cd59b-0272-49d7-926a-57f0181cb3fa" />

**Q2:** For this eample, if memory is sufficient and native_granularity is [128, 128], can I fuse all the nodes and use granularity [128, 128, 32] ?

**Q3:** If Q2 is feasible, Is the computing time for a single turn calculated as $2000 \cdot \frac{256}{128} + 2000 \cdot \frac{32}{256} = 4250$?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-20T23:44:20Z
- url: https://github.com/yarongmu-google/MLSys/issues/74#issuecomment-4284965010

<comment>
Thanks for the question.

Re Q1: no, the chosen granule can't exceed their native counterparts. Think about the native granule as the hardware capabilities - a hardware comes with a given size, and we can't make it bigger just because we have the need.

Re Q2: yes, you can, assuming 32 is smaller than the native granule (note that the native k granule is always the same as the w/h granules), assuming enough memory (I didn't verify this).

Re Q3: compute time for a single turn is based on the native granule: smaller w and h are padded, so they always consume the full cost; however, k is streamed, so it's proportional to the k granule. Thus, your cost formula is not correct. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-20T23:44:47Z
- url: https://github.com/yarongmu-google/MLSys/issues/74#issuecomment-4284966603

<comment>
I will resolve this for now. Pleas create a new issue, referencing this one,  if the above doesn't make sense.
</comment>

---

