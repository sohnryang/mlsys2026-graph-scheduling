---
source: github
repo: owner/repo
issue_number: 71
issue_title: "Validity of Split-K, Revisited"
issue_url: https://github.com/yarongmu-google/MLSys/issues/71
exported_at: 2026-04-23T09:44:14Z
---

# Issue #71: Validity of Split-K, Revisited

## Original post
- author: sohnryang
- created_at: 2026-04-07T14:51:25Z
- url: https://github.com/yarongmu-google/MLSys/issues/71

<comment>
> Thanks for the question.
> 
> #63 was about pointwise -> (split-k) matmul. However, the question here is about (split-k) matmul -> pointwise. These two are completely different situations. As for when split-k can be applied, please use linear algebra to decide. 
> 
> Re the validator, like I explained before, may rule out innovative scheduling. An example is the famous flash attention, which was not in anyone's "official" scheduling playbook until it's discovered. 

 _Originally posted by @yarongmu-google in [#67](https://github.com/yarongmu-google/MLSys/issues/67#issuecomment-4196030196)_

I don't think this clarification is sufficient. As far as I know, linear algebra doesn't precisely specify what is valid for split-k execution (as it's an implementation detail rather than math), let alone dictate what kind of pointwise and matmul fusion is possible. For example, "using linear algebra to decide" is insufficient to rule out the option 1 (recomputation) in #63's point 2.
</comment>

---

## Comment 1
- author: natetyoung
- created_at: 2026-04-07T18:37:40Z
- url: https://github.com/yarongmu-google/MLSys/issues/71#issuecomment-4201373738

<comment>
Additionally, there seems to be some confusion: my original question #63 was about pointwise -> (split-k,split-w,split-h) matmul, but the answer stated that the fusion was invalid because the pointwise could not participate in the split-k iteration. I don't see how this is the case; a pointwise whose output is the first input of a matmul operates on the h and k dimensions of that matmul, and so could participate in split-k. Is the issue that it couldn't participate in the split-w dimension?
</comment>

---

## Comment 2
- author: papp-pal-andras
- created_at: 2026-04-10T15:30:54Z
- url: https://github.com/yarongmu-google/MLSys/issues/71#issuecomment-4224856929

<comment>
I agree that the answer to #63 was very surprising in several ways, or at least seems to contradict the spirit of "not ruling out innovative scheduling" discussed in other answers.

1. To my understanding, a split-k only affects the execution of the sink operator(s), right? Their input operators in the subgraph do not use split-k anymore, since then there would be no "unified execution grid", we'd have different number of iterations for different operators. See Example 5 startegy B, where Op[1] uses split-k tiling, but its input Op[0] does not use split-k tiling over Tensor[0] and Tensor[1].
The example in issue #63 seems identical to Example 5 strategy B, it's just that Op[0] is pointwise now. The answer to #63 says that the "The entire subgraph executes at a single granularity", but this is as true for a Pointwise Op[0] as the MatMul Op[0] in Example 5: neither of them can participate in a split-k directly because they are not sink operations, but both can be executed in the same way as per linear algebra rules.

2. If you maintain that the answer to #63 is correct, would the subgraph still be invalid with spatial tile size 256x256, i.e. if there were only 2 iterations, just according to the split-k dimension? There's no repetition of tiles in Tensor0 (the pointwise's input) now, but of course Op0 still cannot be split-k tiled since it is Pointwise.

3. Finally, a detour regarding the original question of #63. If I understood the intent of the question right, this can occur also without any split-k or pointwise operators. In a simple chained MatMul like Example 5 (let's assume infinite fast memory, native granularity [64, 64]), with subgraph [0,1], execution granularity [64, 64, 128], traversal order [0,2,1,3], the situation is essentially the same: both row strips of Tensor[3] are computed twice via Op[0]. To my understanding, the answer to #63 is then option 1, the computation happens twice. Is that correct?
</comment>

---

## Comment 3
- author: PKU-DwQ
- created_at: 2026-04-17T06:18:34Z
- url: https://github.com/yarongmu-google/MLSys/issues/71#issuecomment-4265868723

<comment>
@papp-pal-andras gave a very helpful summary of the points of confusion that some of us may still have about the rule.
It would extremely helpful if @yarongmu-google could provide examples and explaination of:
1. mixed fusion of matmul + pointwise ops
2. fusion involving matmuls larger than the native granularity

For (2), in example 5 the tile sizes are smaller than the native granularity, so I'm not sure how to interpret the statement: "The entire subgraph executes at a single granularity." 
</comment>

---

## Comment 4
- author: yarongmu-google
- created_at: 2026-04-22T03:37:02Z
- url: https://github.com/yarongmu-google/MLSys/issues/71#issuecomment-4293344525

<comment>
Fair pushback — "use linear algebra to decide" was too terse. Let me lay out the rule explicitly and walk through the specific cases raised.

### Two governing rules

**1. Commutation.** Pointwise is element-wise, so it commutes with gather-splits (the output-partitioning splits along h and w) but does **not** commute with the reduce-split (the work-partitioning split along K). `P(X₁ + X₂) ≠ P(X₁) + P(X₂)` unless `P` is linear. Concretely: a pointwise never consumes a reduce-split partial sum.

**2. Per-tile execution.** Directly from the spec: *"For Pointwise operations, since they have no reduction dimension, k is ignored (effectively treating it as 1), and the operation simply executes **once per spatial tile**."* Pointwise does not ride a matmul's per-k-step loop. When a pointwise fires for one of its own `(w × h)` output tiles, the data feeding it must be fully formed at that moment — not a partial accumulator.

From these two together: **pointwise is valid at position X iff the tensor it consumes (or produces, for a prologue) is a fully-formed tile at that point, not a reduce-split partial.**

### Applying the rule

**@sohnryang — "linear algebra is insufficient."** Agreed. Commutation constrains which algebraic rearrangements are allowed; per-tile execution (rule 2) pins down the implementation validity on top. The combination is what you need.

**@natetyoung — "pointwise whose output is a matmul LHS operates on (h, k), so why can't it participate in split-k?"** The dimensional argument is correct, but the obstacle is the tile shapes under the shared `[w, h, k]` grid. Pointwise tiles at `(h × w)`. The matmul reads LHS strips of shape `(h × k)` per k-step. For the matmul's k-loop at any row to read its strips from resident pointwise data, the pointwise tile at that row must have already produced the full matmul K-axis worth of LHS — i.e. a single pointwise tile must span the whole K dimension of the matmul. That fixes the granule condition:

- Pointwise feeding a matmul's LHS → require `w ≥ matmul.K`
- Pointwise feeding a matmul's RHS → require `h ≥ matmul.K`

When this holds, one pointwise tile produces the full LHS (or RHS) row strip, and the matmul's per-k-step strip reads land on a slice of that already-resident tile — no reload, no partial sums, compute-neutral. When it fails, the matmul's k-loop asks for data the pointwise hasn't produced yet under the per-tile execution rule. So the issue is geometric, not specifically about split-w.

**@papp-pal-andras — three sub-questions.**

*Q1 ("split-k only affects the sink operator"):* In a pure matmul chain with outer split-k (the pattern illustrated in Example 5 Strategy B), yes — the inner matmul produces a true `(h × k)` gather-strip per outer k-step, compute-neutral. For pointwise→matmul the reasoning doesn't carry over unchanged because pointwise has no k-axis tile. Its grid is `(w × h)`; it only produces full tiles. The valid escape hatches are the granule-alignment condition in the @natetyoung answer above.

*Q2 ("256×256 spatial tile, only 2 iterations from k=128, no spatial repetition"):* This configuration **is valid**. With `w = 256 ≥ matmul.K = 256`, the pointwise produces the full `256 × 256` LHS in a single tile; the matmul's two k-steps read `256 × 128` RHS strips against the resident LHS; fast memory holds `pointwise_output + rhs_strip + accumulator`. If that fits, the plan is compute-neutral and the roofline cost is accurate. The #63 verdict applied to the specific small-tile geometry in the original question (`w, h < K`) where the alignment condition fails. It was stated too broadly — the correct rule is the granule-alignment condition, not a blanket "no split-k in mixed subgraphs." Sorry about the confusion that caused.

*Q3 ("chain-2, granule [64,64,128], traversal [0,2,1,3]: computed twice?"):* This surfaces a real spec ambiguity about whether intra-subgraph ephemerals persist across tile iterations. Under a strict "ephemerals never persist across tiles" reading, the intermediate's row-strip is re-produced every time the traversal revisits a row — so yes, "computed twice" in your 2×2 example. Under a more permissive reading (intra-subgraph residency managed by the hardware's cache across tiles, subject to working-set capacity), the strip caches and isn't recomputed. The chain-2 outer-split-k pattern's compute-neutrality relies on the permissive reading; under the strict reading it would pay an `N₂/w` penalty. The spec doesn't pin this down — deserves its own thread to nail down. I'll open one.

**@PKU-DwQ — "mixed fusion examples; matmul larger than native":**

*Mixed fusion.* A valid prologue pattern looks like: pick `w = matmul.K` (or larger), run the pointwise once to produce the full LHS row-strip, then let the matmul do its k-loop against the resident LHS. The `w = matmul.K` choice is the canonical case. An epilogue (matmul → pointwise) is always valid regardless of granule: the matmul's k-loop finishes, producing a fully-formed output tile; the pointwise then runs on it. The invalid case is a fused plan where the pointwise tile doesn't cover the matmul's K-axis and the full pointwise output doesn't fit in fast memory either — the matmul's k-loop would ask for data that hasn't been produced under per-tile execution.

*MatMul larger than native.* "Single granularity" refers to the chosen `[w, h, k]` tuple applying uniformly to all ops in the subgraph, not to each op fitting in a single tile. Tensors larger than the granule simply tile under it — the subgraph iterates over `⌈M/h⌉ × ⌈N/w⌉` spatial tiles per matmul. The subgraph's "execution unit" is one granule-sized tile; the subgraph runs it as many times as the tensor shape requires.

### Summary of the corrected rule

Fusion validity for a mixed MatMul + Pointwise subgraph with `k < max_K`:

- **Epilogue** (matmul → pointwise): always valid.
- **Prologue / between, where a pointwise produces a matmul's LHS**: requires `w ≥ matmul.K` (so a single pointwise tile covers the whole K-axis of the LHS).
- **Prologue / between, where a pointwise produces a matmul's RHS**: requires `h ≥ matmul.K`.
- Otherwise: not realizable under per-tile execution, and submissions that assume the hardware rescues this via hidden recomputation are not pricing it correctly.

Thanks to everyone who pushed on this. The earlier answers were narrower than the general rule warranted.
</comment>

---

## Comment 5
- author: yarongmu-google
- created_at: 2026-04-22T03:37:25Z
- url: https://github.com/yarongmu-google/MLSys/issues/71#issuecomment-4293345664

<comment>
I will resolve this for now. Please file a new issue, referencing this one, if the above doesn't make sense.
</comment>

---

