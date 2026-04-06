---
source: github
repo: owner/repo
issue_number: 58
issue_title: "The Granularity of Multiple Outputs"
issue_url: https://github.com/yarongmu-google/MLSys/issues/58
exported_at: 2026-04-06T10:36:38Z
---

# Issue #58: The Granularity of Multiple Outputs

## Original post
- author: gychen233
- created_at: 2026-03-25T10:43:44Z
- url: https://github.com/yarongmu-google/MLSys/issues/58

<comment>
I fully thank the organizers for addressing our comments, but I still have a lot of confusion.

As an extension of #20 #28.

I would like to clarify the definition of 'incompatible granularity ' when grouping operations. 

Suppose two operations in a subgraph produce output matrices of sizes 256x256 and 128x256, respectively. Can I use an execution granularity of 128x128 and simply mask/silently **ignore** the execution when the spatial grid exceeds the boundaries of the smaller matrix? 

Additionally, is it allowed for a single subgraph to have **multiple output operations** of **different types**—specifically, one being a MatMul and the other a Pointwise operation?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T04:25:01Z
- url: https://github.com/yarongmu-google/MLSys/issues/58#issuecomment-4190289942

<comment>
Thanks for the question.

- Suppose two operations in a subgraph produce output matrices of sizes 256x256 and 128x256, respectively. Can I use an execution granularity of 128x128 and simply mask/silently ignore the execution when the spatial grid exceeds the boundaries of the smaller matrix? -> Yes, this is valid. The execution granularity [w, h, k] sets the tile size, not the grid dimensions. Different operations in the same subgraph may have different tile counts based on their output sizes; each operation's output is fully tiled and computed. This is standard output-stationary scheduling, as shown in several PROBLEM.md examples.

- Additionally, is it allowed for a single subgraph to have multiple output operations of different types—specifically, one being a MatMul and the other a Pointwise operation? -> yes, as long as your scheduling produced all results at least once.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T04:25:23Z
- url: https://github.com/yarongmu-google/MLSys/issues/58#issuecomment-4190290681

<comment>
I will resolve this for now. Please open a new issue, reference this one, if the above doesn't make sense.
</comment>

---

