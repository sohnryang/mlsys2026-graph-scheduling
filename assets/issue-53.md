---
source: github
repo: owner/repo
issue_number: 53
issue_title: "Pointwise Operation Dimension Mismatch in Benchmark 13"
issue_url: https://github.com/yarongmu-google/MLSys/issues/53
exported_at: 2026-03-21T09:12:50Z
---

# Issue #53: Pointwise Operation Dimension Mismatch in Benchmark 13

## Original post
- author: Wimage3141
- created_at: 2026-03-18T22:41:46Z
- url: https://github.com/yarongmu-google/MLSys/issues/53

<comment>
There seems to be a dimension mismatch in three pointwise operations: Op[48], Op[49], and Op[50]
For example, Op[48] (49th operation) has:
T[36], T[39] → T[82]
[128×128], [128×128] → [128×4096]


Here's the script I used for finding this:
```
violations = []

for k, op in enumerate(op_types):
    if op != "Pointwise":
        continue
    all_tensors = inputs[k] + outputs[k]
    dims = [(heights[t], widths[t]) for t in all_tensors]
    if len(set(dims)) > 1:
        violations.append(k)
        print(f"k = {k}")
        print(f"  Inputs  = {inputs[k]} -> {[(heights[t], widths[t]) for t in inputs[k]]}")
        print(f"  Outputs = {outputs[k]} -> {[(heights[t], widths[t]) for t in outputs[k]]}")
if not violations:
    print("all Pointwise ops have matching dims.")
```
</comment>

---

## Comment 1
- author: tonibohnlein
- created_at: 2026-03-19T11:02:35Z
- url: https://github.com/yarongmu-google/MLSys/issues/53#issuecomment-4089340985

<comment>
 There seem to be more bugs in benchmark 13. Tensors 97, 98, 99 are defined but no operation produces or consumes them. 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-03-20T03:14:59Z
- url: https://github.com/yarongmu-google/MLSys/issues/53#issuecomment-4095158175

<comment>
Thanks for the report, @Wimage3141! You're right — Tensors 82 and 83 in benchmark 13 were incorrectly declared as 4096x128 instead of 128x128. This has been fixed: both tensors now have dimensions consistent with their Pointwise operations. No other benchmarks are affected.                                                                      
                                                                                                                       
@tonibohnlein — Tensors 97, 98, 99 being unused is fine; the constraint is that there are no orphaned operations,  not tensors. Unused tensors are simply ignored. 
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-03-20T03:15:31Z
- url: https://github.com/yarongmu-google/MLSys/issues/53#issuecomment-4095159988

<comment>
I will resolve this for you now. Please reopen if the above doesn't make sense.
</comment>

---

