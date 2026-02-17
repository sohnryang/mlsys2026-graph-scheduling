---
source: github
repo: owner/repo
issue_number: 3
issue_title: "Matmul Tile Reuse - Example 4"
issue_url: https://github.com/yarongmu-google/MLSys/issues/3
exported_at: 2026-02-17T08:42:11Z
---

# Issue #3: Matmul Tile Reuse - Example 4

## Original post
- author: xavierrouth
- created_at: 2026-02-04T17:40:01Z
- url: https://github.com/yarongmu-google/MLSys/issues/3

<comment>
Hi, I do not understand the Tile reuse rules, as described in Example 4. 

In Strategy A, Steps 2 and 4 should reuse tiles loaded by Steps 1 and 3. I.e: 
- step 2 should reuse row strip 0 loaded by step 1.
- step 4 should reuse row strip 0 loaded by step 3.

This is currently not included in the latency calculation for Strategy A. With reuse, Step 2 and 4 both are compute bound, and take 1,500 cycles each (as in Strategy B). This makes the total Strategy A latency 7,096 cycles, but the current latency in the document is 8,192 cycles.
</comment>

---

## Comment 1
- author: ericxu233
- created_at: 2026-02-05T06:32:12Z
- url: https://github.com/yarongmu-google/MLSys/issues/3#issuecomment-3851341384

<comment>
I am also wondering about this.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T19:44:57Z
- url: https://github.com/yarongmu-google/MLSys/issues/3#issuecomment-3868083343

<comment>
Thanks for the question. 

The calculation is correct as written.

ou are absolutely right—your proposed schedule is a valid and more efficient solution (than Strategy A).

However, the strategies listed in the Problem Description are not intended to be an exhaustive list of all valid solutions. Instead, they are specific demonstrations chosen to illustrate how different scheduling decisions (like memory management and tiling) impact the latency calculation.

Strategy A was explicitly chosen to demonstrate a "naive reload" scenario. We wanted to show exactly what happens when you don't reuse data: you pay the full bandwidth penalty. The calculation (8,192) is correct for that specific, suboptimal choice.

Your observation describes a more optimized strategy where you do utilize tensors_to_retain to avoid those reloads. That would indeed result in the lower latency you calculated (7,096).

So, while your solution is better, the example in the text is factually correct for the "dumb" schedule it is depicting. We use these contrasting examples to help participants understand the cost mechanics they will need to optimize.


</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-02-08T19:45:12Z
- url: https://github.com/yarongmu-google/MLSys/issues/3#issuecomment-3868084765

<comment>
I will resolve this for now. Please reopen if the above doesn't make sense. 
</comment>

---

