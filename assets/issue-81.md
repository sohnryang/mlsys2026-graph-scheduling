---
source: github
repo: owner/repo
issue_number: 81
issue_title: "Follow-up questions of #71"
issue_url: https://github.com/yarongmu-google/MLSys/issues/81
exported_at: 2026-04-23T09:45:35Z
---

# Issue #81: Follow-up questions of #71

## Original post
- author: gychen233
- created_at: 2026-04-22T14:00:01Z
- url: https://github.com/yarongmu-google/MLSys/issues/81

<comment>
The answer in #71  is very reasonable. I comprehend that the granularity apply to all the operators in the current subgraph. I want to clarify more details about it.

Q1: Is the single execution size limitation of an operation determined by the native_granularity or the granularity? For example, if the native_granularity is [128, 128] and I set the granularity to [64, 64, 64], can the single execution size of an operator still be 128?

Q2: Is the granularity[2] also uniformly enforced across all operations within the subgraph? For instance, in example #74 , is op0 required to use this specific reduction depth and execute eight times to get the output?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-22T20:36:30Z
- url: https://github.com/yarongmu-google/MLSys/issues/81#issuecomment-4299781558

<comment>
Thanks for the question.

Re Q1: I am not sure what you meant by "single execution size limitation", but perhaps this mental model analogy will help: imagine you want to fill a giant water tank with water, and you can use a small bucket to do so. Here, the size of the water tank is the tensor size (ie your workload size); the small bucket you can use is the native granule (ie your tool size). Your job is to figure out how many trips you will need to make to fill the water tank, and how much you choose to fill the water bucket each trip (ie your chosen granule), perhaps not always 100% full on each trip because you are out of energy (ie the memory constraints). This is not a perfect analogy, but should help you see the relationship among the native granule, the workload size, and your chosen granule. The point is that even if you choose to only fill the bucket 50% of its capacity on one trip, you still have to use the whole physical bucket - you don't get a smaller bucket just because you could use a smaller bucket, and you certainly don't get a larger bucket just because you want to fill it 200%. Hope this helps.

Re Q2. The spec made this clean IMHO: "The reduction dimension (k) is streamed temporally through the array - choosing k below native simply runs fewer cycles, dividing compute proportionally without waste." That's the key difference from spatial - no padding penalty, no waste. And per the unified-grid rule, the chosen k applies to every op in the subgraph. As for the example in #74: it didn't say if op) is a matmul or pointwise. If matmul, yes Op0 will have to use the reduction depth; else, k is not applicable.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-22T20:36:54Z
- url: https://github.com/yarongmu-google/MLSys/issues/81#issuecomment-4299783760

<comment>
I will resolve this for now. Please open a new issue, referencing this one, if the above doesn't make sense.
</comment>

---

