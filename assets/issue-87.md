---
source: github
repo: owner/repo
issue_number: 87
issue_title: "Final clarification on unified execution grid"
issue_url: https://github.com/yarongmu-google/MLSys/issues/87
exported_at: 2026-04-25T01:17:30Z
---

# Issue #87: Final clarification on unified execution grid

## Original post
- author: natetyoung
- created_at: 2026-04-24T21:35:06Z
- url: https://github.com/yarongmu-google/MLSys/issues/87

<comment>
I don't think it'll affect my solution at this late date, but I wanted to ask for a final clarification on the notion of a unified execution grid.

From the examples and from previous clarifications, there are 2 things it might mean, as I see it:

1. "Unified Execution Grid" could mean something like "all operations in the subgraph must be fused at the same level, and the tile size of ops which are not subgraph outputs is derived from the execution grid according to what data is necessary for each output tile to be computed." This was my initial understanding, and it seems to be how Example 5, strategy B works. `Op[1]` has the stated granularity of `w=128, h=128, k=32`; this means that at each tile, it requires a `w=32, h=128` strip of `Tensor[3]`, so `Op[0]` must be executed at a "local" granularity of `w=32, h=128, k=128` in order to produce that in a single tile. This interpretation also made the (later declared to be erroneous) constraint expressed in #32 make a little more sense: although it is physically possible to accumulate the result of the matmul in many k-steps and only then run the pointwise, the unified execution grid rule forbids using more than a single tile of one op to produce the data for a single tile of a different op.
2. "Unified Execution Grid" could instead mean something like "the tile size in the w, h, and k directions must be the same for every op in its own coordinate space, regardless of the relationships between sizes among the ops". This seems to be what the answer to #71 is saying: a pointwise->matmul LHS fusion with matmul granularity `[w, h, k]` would only _physically_ require that a `h x k` tile of the matmul LHS input is produced by the pointwise op at each iteration, but the unified execution grid rule requires that the `w` and `h` of the matmul are the same as the _local_ `w` and `h` of the pointwise, and because the pointwise does not step in `k`, it must produce the entire `K_full` of the matmul in one shot, thus `w >= K_full`.

Which of these, if either, is correct? I believe this is where the confusion expressed in #84 came from: #71 seems to imply that the ("local") `k` granularity for each subgraph op must be the same, but Example 5 strategy B doesn't show it.

For my part, my team will submit a solver which is a little conservative about fusion in order to abide by something like the union of these two rules (without the erroneous constraint from #32), but just for confirmation, I'd really like to know which interpretation is correct.

Also, if I can piggyback on this for two simpler questions: 1. should we put all team member names under "full name of submitter" on the submission form, or will there be another place to put it? and 2. the readme says the zip file submission deadline is today and the writeup deadline is in a week, but then says the zip file should include a writeup.pdf. Do we actually need to submit a writeup today, and/or will there be a separate submission process later?

Thanks again for fielding all these questions.
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-24T22:17:18Z
- url: https://github.com/yarongmu-google/MLSys/issues/87#issuecomment-4316731217

<comment>
Thanks for the question. The 2 interpretations you listed are just different ways of saying the same thing. 

Think about the "Unified Execution Grid"  as output stationary - when you need to produce a giant tensor, you need to tile the large workload into <= native granule pieces. Once you have the output fixed, the input slices etc will be determined by that. Example 5(B), for example, shows that when [h, w] is not tiled, but only k-is tiled - that means eg Tensor1 needs to be sliced vertically, providing column-like inputs. #71 makes it sound like sth diff, because that's to address the specific pointwise + matmul fusion issues.

So one thing that's beyond this specific problem that I observed (I could be wrong) based on all these questions: while the problem focuses on scheduling - ie find a scheduler, not reconstruct the underlying hardware behavior (I think this causes a lot of speculations of how the hw works exactly) or find exotic fusion opportunities. This problem tries to promote instead the concept for one type of "pipelining". Ie, roughly speaking, aggressive fusion is often the most beneficial when the fast memory is highly constrained, while pipelining applies more broadly across different types of hardware. More specifically, if one tries both, one would see that in almost all benchmarks, as long as the scheduler is generally competent in identifying subgraphs that can be properly pipelined, the results are both good and stable, across diff workload shapes.

As for why the hw is abstracted this way w/o further lower-level details, the goal is to come up with novel schedulers that work across hw w/ diff lower-level implementation details but all share this same abstraction, w/o tying to any one of them. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-24T22:17:45Z
- url: https://github.com/yarongmu-google/MLSys/issues/87#issuecomment-4316732911

<comment>
I will resolve this for now. Please open a new issue, citing this one, if the above doesn't make sense.
</comment>

---

