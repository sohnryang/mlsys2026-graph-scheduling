---
source: github
repo: owner/repo
issue_number: 37
issue_title: "MatMul Tile Reuse - Example 4A again"
issue_url: https://github.com/yarongmu-google/MLSys/issues/37
exported_at: 2026-03-17T10:59:47Z
---

# Issue #37: MatMul Tile Reuse - Example 4A again

## Original post
- author: papp-pal-andras
- created_at: 2026-03-11T15:13:28Z
- url: https://github.com/yarongmu-google/MLSys/issues/37

<comment>
Hi! It seems to me that the original question in Issue #3 has still not been answered/resolved. (There were various follow-ups and confusions in Issues #3 and #15 , but let's focus on the original question in #3.)

For Example 4A, the correct latency indeed seems to be 7,096 instead of 8,192. This is just a single subgraph, single operation, all Implicit Reuse, all Intra-Subgraph. The naive tiling order [0, 1, 2, 3] in Strategy A can still reuse row strip 0 from Step 1 to Step 2, and reuse row strip 1 from Step 3 to Step 4. This seems to give a cost of 7,096 (still higher than the zig-zag order [0, 1, 3, 2], but not what the sample solution says).

(The only possible explanation I could think of for the sample solution is that having 'null' in the input instead of [0, 1, 2, 3] somehow prohibits Implicit Reuse, but this seems unlikely; the text specifically says that the "system defaults to Raster order".)

Thanks in advance!
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-03-11T20:08:05Z
- url: https://github.com/yarongmu-google/MLSys/issues/37#issuecomment-4041887415

<comment>
You're absolutely right, and I apologize for the confusion — this was a genuine error in the example. 

As the problem description states, tensors_to_retain controls inter-subgraph persistence only. Intra-subgraph data reuse is managed automatically by the hardware. This means that even in raster order, the hardware keeps input strips resident when consecutive tiles share them.                                 
                                                                                                              
With raster order [0, 1, 2, 3]:
  - Step 1 (top-left): Load row strip 0 + col strip 0 → 2,048                                                 
  - Step 2 (top-right): Reuse row strip 0, load col strip 1 → 1,500
  - Step 3 (bottom-left): Load row strip 1 + col strip 0 → 2,048
  - Step 4 (bottom-right): Reuse row strip 1, load col strip 1 → 1,500
  - Total: 7,096 (2 reuses)

The "flush the fast memory every time" description was wrong — it contradicted our own spec. We've fixed this in https://github.com/yarongmu-google/MLSys/commit/af015aa. Strategy A now correctly shows 7,096, and Strategy B (zig-zag) remains better at 6,548 thanks to 3 reuses instead of 2.

Thank you for catching this, and apologies it took a while to resolve.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-03-11T22:54:49Z
- url: https://github.com/yarongmu-google/MLSys/issues/37#issuecomment-4042752508

<comment>
I will resolve this for you now. Please open new issue if the above doesn't make sense.
</comment>

---

