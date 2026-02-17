---
source: github
repo: owner/repo
issue_number: 13
issue_title: "Number of inputs per op"
issue_url: https://github.com/yarongmu-google/MLSys/issues/13
exported_at: 2026-02-17T08:43:10Z
---

# Issue #13: Number of inputs per op

## Original post
- author: aheirman
- created_at: 2026-02-07T21:33:59Z
- url: https://github.com/yarongmu-google/MLSys/issues/13

<comment>
```
{
  "widths": [128, 128, 128],
  "heights": [128, 128, 128],
  "op_types": ["MatMul", "Pointwise"],
  "inputs": [[0], [1]],
  "outputs": [[1], [2]],
  "base_costs": [100, 10],
  "fast_memory_capacity": 20000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}
```

example_problem.json shows a matmul with a single operand.
ONNX does not permit this. 
Is this intentional?
Could we have the specification of the operands?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T18:06:34Z
- url: https://github.com/yarongmu-google/MLSys/issues/13#issuecomment-3867777765

<comment>
Thanks for flagging this! You are absolutely correct.

That was indeed a typo in our example file. I have just pushed a fix (commit b2317e6) that updates example_problem.json to have the first operation as Pointwise too. (This was supposed to be the json for Example 1 in the write up but obviously I made some changes and forgot top update here - sorry).

Appreciate the help in catching this early!
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T18:27:45Z
- url: https://github.com/yarongmu-google/MLSys/issues/13#issuecomment-3867826868

<comment>
I will close this for now. Please reopen if the above doesn't make sense :) 
</comment>

---

