---
source: github
repo: owner/repo
issue_number: 11
issue_title: "example 5b why do we move the entire Tensor0 in step 1"
issue_url: https://github.com/yarongmu-google/MLSys/issues/11
exported_at: 2026-02-17T08:42:58Z
---

# Issue #11: example 5b why do we move the entire Tensor0 in step 1

## Original post
- author: aheirman
- created_at: 2026-02-07T16:44:27Z
- url: https://github.com/yarongmu-google/MLSys/issues/11

<comment>
granularity definition 
```
For MatMul inputs, the Left-Hand Side (LHS) input requires width k (reduction depth) and height h, while the Right-Hand Side (RHS) Input requires width w and height k.
```

Example 4A
```
  "granularities": [[64,64,128]],
Step 1 (top-left):

    Move row strip 0 from the slow memory to the fast memory
```

Example 5b
```
"granularities": [[128, 128, 32]],

Step 1 (k=0..31): 


    Move Tensor0 (128x128) from slow memory. MemoryTime1_in = (128x128)/10 = 1638.4
```
Why do we move the entire Tensor0 in step 1 of Example 5b?

</comment>

---


