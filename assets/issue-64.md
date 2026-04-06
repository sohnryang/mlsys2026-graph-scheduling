---
source: github
repo: owner/repo
issue_number: 64
issue_title: "Limitations of Ephemeral Data"
issue_url: https://github.com/yarongmu-google/MLSys/issues/64
exported_at: 2026-04-07T10:36:32Z
---

# Issue #64: Limitations of Ephemeral Data

## Original post
- author: Richard1688Sun
- created_at: 2026-04-06T04:52:41Z
- url: https://github.com/yarongmu-google/MLSys/issues/64

<comment>
I have a hard time understanding the hardware setup for ephemeral data.

If we had the following operations:
T1 -> Op1 -> T2
T2 -> Op2 -> T3
T2 -> Op3 -> T4

And in 1 subgraph we fuse all 3 of them, is T2 ephemeral? T2 is intermediate data, however there are 2 operations that need T2. On an actual hardware, where would T2 be stored if it never touches fast memory? Especially since the Compute Cost within 1 subgraph is additive that suggests each operation within the subgraph is done 1 after another.

Any clarification on this would be greatly appreciated!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-06T15:51:05Z
- url: https://github.com/yarongmu-google/MLSys/issues/64#issuecomment-4193294242

<comment>
Thanks for the question.

In your example, T2 is ephemeral in the context of this competition, since we assume that all ops within the same subgraph is fused together.

On an actual hardware, T2 would have been temporarily stored on the compute unit's register. So you are right that removing this makes the problem less intuitive; however, we did that to (1) decrease the difficulty, so that participants do not have to model a 3-level memory system; (2) on the actual GPU/TPU hardware, the registers are managed by the compilers, which automatically spills if the register doesn't have enough space, so you can roughly think they have unlimited memory (subject to their spill space limitation but we removed this whole concept to make the abstraction tighter).
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-06T15:51:33Z
- url: https://github.com/yarongmu-google/MLSys/issues/64#issuecomment-4193296389

<comment>
I will resolve this for now. Please open a new ticket, reference this one, if the above doesn't make sense.
</comment>

---

