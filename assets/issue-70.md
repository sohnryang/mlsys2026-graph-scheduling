---
source: github
repo: owner/repo
issue_number: 70
issue_title: "Tensor reuse if there was a full tensor and a partial tensor"
issue_url: https://github.com/yarongmu-google/MLSys/issues/70
exported_at: 2026-04-23T09:44:04Z
---

# Issue #70: Tensor reuse if there was a full tensor and a partial tensor

## Original post
- author: Richard1688Sun
- created_at: 2026-04-07T02:19:28Z
- url: https://github.com/yarongmu-google/MLSys/issues/70

<comment>
Extending off #65 

What would happen within a subgraph, if 1 operation loads a full `T0` but another operation only requires a strip of `T0`? Would this be intra-subgraph reuse or will we need to double count the T0 strip?

Does the situation change if T0 was fully retained from the previous subgraph? What decides the tensor renaming when fetching from slow memory?

Thanks for all the clarification!
</comment>

---

## Comment 1
- author: papp-pal-andras
- created_at: 2026-04-10T15:32:22Z
- url: https://github.com/yarongmu-google/MLSys/issues/70#issuecomment-4224867591

<comment>
Good question, I'm also interested in this one!
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-21T23:30:55Z
- url: https://github.com/yarongmu-google/MLSys/issues/70#issuecomment-4292531916

<comment>
Thanks for the  questions. The rule from #65 applies cleanly here; let me restate it so it is explicit.

**The naming rule (asymmetric):**
- **Full-tensor move → superset name.** If T0 arrives in fast memory as a full tensor — either because some op loaded it whole, or because it was retained at tensor granularity — any subsequent access (full *or* partial) is a hit. No reload.
- **Partial move → opaque slice-specific name.** A strip load creates a name tied to the exact slice-spec (e.g. `(T0, row=r)` for a MatMul LHS row-strip). Another partial only hits on **exact slice-spec match**; overlapping partials do not reuse. This is the "partial reuse is not feasible" clause.

This asymmetry means you never have to reason about partial-partial overlaps — by construction they don't reuse, regardless of containment.

**Answers to your questions:**

> Intra-subgraph reuse or double count?

*Intra-subgraph reuse* — T0 is charged once at full size. The pointwise/mixed cost model loads T0 on first touch (as a full tensor, since pointwise needs full `w × h` tiles across its output), and the matmul's strip access rides the resident full copy for free.

> Does the situation change if T0 was fully retained from the previous subgraph?

Same outcome, cleaner reason: retention is tensor-granularity, so T0 is resident under the full name at subgraph entry. Both the full-consumer and the strip-consumer hit — no load cost and no retention-byte re-charge.

> What decides the tensor renaming when fetching from slow memory?

The **kind of move** decides:
- Whole-tensor move → canonical name (covers any later full or partial access).
- Partial move → slice-spec-specific name (covers only identical-slice-spec accesses).

This matches a common compiler buffer-allocation model: a buffer is sized to exactly what was moved, and the hardware indexes into it for sub-reads — but a separate partial move allocates a separate buffer.
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-04-21T23:31:17Z
- url: https://github.com/yarongmu-google/MLSys/issues/70#issuecomment-4292532990

<comment>
I will resolve this for now. Please create anew issue, citing this one, if the above doesn't make sense.
</comment>

---

