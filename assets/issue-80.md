---
source: github
repo: owner/repo
issue_number: 80
issue_title: "Followup on recent clarifications about native_granularity"
issue_url: https://github.com/yarongmu-google/MLSys/issues/80
exported_at: 2026-04-23T09:45:30Z
---

# Issue #80: Followup on recent clarifications about native_granularity

## Original post
- author: natetyoung
- created_at: 2026-04-22T05:51:24Z
- url: https://github.com/yarongmu-google/MLSys/issues/80

<comment>
Yesterday, the answer to #74 and the associated change to PROBLEM.md stated that `native_granularity` applies to `k`, as well as `w` and `h`. I find this extremely surprising, and have three major questions:

1. The `native_granularity` given in the benchmarks only has two numbers (perhaps by chance, they are always `[128, 128]` in the publicly released benchmarks). If it applies only to `w` and `h`, this makes sense. If it also applies to `k`, and the two numbers are allowed to differ, which is the `k` granularity?
2. Is the `base_cost` for a matmul just the cost for the native reduction depth, or the entire reduction depth? PROBLEM.md seems to imply that it is the native depth, but the answer to #27 seems to say it is the entire depth.
3. Does this mean that, in a benchmark with `native_granularity = [128, 128]`, a MatMul with reduction size greater than 128 (take, for instance, any matmul in benchmark 1) must **_always_** be split in `k`? If so, does this mean that a matmul like that can never experience intra-operator reuse of its inputs (since in order to continue the reduction in output-stationary mode, input slices must always step in `k`) and can never be fused with an operation which consumes its output (since the output takes multiple iterations before it is fully accumulated for a single spatial tile)?
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-22T20:52:53Z
- url: https://github.com/yarongmu-google/MLSys/issues/80#issuecomment-4299898892

<comment>
Thanks for the questions.

Re Q1: Per PROBLEM.md line 33: "the array's streaming depth in the reduction dimension equals its spatial extent, so native_granularity: [128, 128] means the native w, h, and k are all 128." -> this is true across all examples, released and unreleased benchmarking datasets. But perhaps I should have explicitly used only 1 number for all 3 native granule dimensions. My bad here.

Re Q2: `base_cost` is the cost per native-spatial-tile covering the native granule's entire reduction depth. Total compute = ⌈M/native⌉·⌈N/native⌉·base_cost; per k-step = base_cost · k/K; sum over K/k steps = base_cost (compute-neutral split-k, per PROBLEM.md's "dividing compute proportionally without waste"). Let me append to #27 to make it clear that "a single operation at the native granularity with the full reduction dimension" may better be "a single operation at the native granularity with the full *native* reduction dimension".

Re Q3: Yes, K > native_k forces split-k. No, that doesn't preclude fusion or reuse — see #82. Because an op lives in one subgraph, the subgraph wraps all the k loops, and the accumulator is fully formed per spatial tile before any consumer fires. Example 5 Strategy B demonstrates intra-op input reuse (Tensor0 resident across all 4 k-steps); tensors_to_retain + traversal_orders are the levers.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-22T20:53:12Z
- url: https://github.com/yarongmu-google/MLSys/issues/80#issuecomment-4299900860

<comment>
I will resolve this for now. Please open a new issue, citing this one, if the above makes sense.
</comment>

---

## Comment 3
- author: natetyoung
- created_at: 2026-04-22T21:47:15Z
- url: https://github.com/yarongmu-google/MLSys/issues/80#issuecomment-4300172242

<comment>
@yarongmu-google Thanks for the quick response. Just to be extra certain about question 2: take op0 in benchmark 1. It will need at least (512/128)^3=64 iterations at full native granularity. In this case, would the _total_ compute cost for **_all_** iterations be 2000*64 (pay once for each iteration in all 3 axes) or 2000*16 (pay once for each iteration in just spatial axes, the reduction axis is already included in the 2000)?
</comment>

---

