---
source: github
repo: owner/repo
issue_number: 85
issue_title: "Recomputation of MatMul"
issue_url: https://github.com/yarongmu-google/MLSys/issues/85
exported_at: 2026-04-25T01:17:13Z
---

# Issue #85: Recomputation of MatMul

## Original post
- author: sohnryang
- created_at: 2026-04-24T14:28:59Z
- url: https://github.com/yarongmu-google/MLSys/issues/85

<comment>
> *Q3 ("chain-2, granule [64,64,128], traversal [0,2,1,3]: computed twice?"):* This surfaces a real spec ambiguity about whether intra-subgraph ephemerals persist across tile iterations. Under a strict "ephemerals never persist across tiles" reading, the intermediate's row-strip is re-produced every time the traversal revisits a row — so yes, "computed twice" in your 2×2 example. Under a more permissive reading (intra-subgraph residency managed by the hardware's cache across tiles, subject to working-set capacity), the strip caches and isn't recomputed. The chain-2 outer-split-k pattern's compute-neutrality relies on the permissive reading; under the strict reading it would pay an `N₂/w` penalty. **The spec doesn't pin this down — deserves its own thread to nail down. I'll open one.**

 _Originally posted by @yarongmu-google in [#71](https://github.com/yarongmu-google/MLSys/issues/71#issuecomment-4293344525)_ (emphasis mine)

It's ~16h before the deadline and no clarification is given on this.
</comment>

---

## Comment 1
- author: sohnryang
- created_at: 2026-04-24T14:49:35Z
- url: https://github.com/yarongmu-google/MLSys/issues/85#issuecomment-4314065238

<comment>
> Apologies for the whiplash and for the loose "always" phrasing in the [#71](https://github.com/yarongmu-google/MLSys/issues/71) summary that made these look inconsistent when the [#71](https://github.com/yarongmu-google/MLSys/issues/71) body had already derived the right rule. **A unified semantics note is coming so this gets stated once, in one place.**

 _Originally posted by @yarongmu-google in [#82](https://github.com/yarongmu-google/MLSys/issues/82#issuecomment-4299544989)_ (emphasis mine)

Also, I can't find the "unified semantics note" as well.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-24T18:06:23Z
- url: https://github.com/yarongmu-google/MLSys/issues/85#issuecomment-4315294217

<comment>
Thanks for the questions.

Please feel free to use the more permissive reading.  Note that your solution is scored not against a rigid source of truth, but based on ablation backed normalization across all submissions, this way ambiguities and potentially different interpretations are taken care of by the scoring system. So whether you use one reading or not doesn't change much of the final result.  #79 contains more details. 

Re the semantics note - that's a hallucination. My apology. 


</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-04-24T18:06:52Z
- url: https://github.com/yarongmu-google/MLSys/issues/85#issuecomment-4315297228

<comment>
I will resolve this for now. Please open a new issue, citing this one, if the above doesn't make sense.
</comment>

---

