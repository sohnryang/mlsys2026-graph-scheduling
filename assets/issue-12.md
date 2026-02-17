---
source: github
repo: owner/repo
issue_number: 12
issue_title: "Tensors created and consumed inside a single subgraph cause ambiguous solutions"
issue_url: https://github.com/yarongmu-google/MLSys/issues/12
exported_at: 2026-02-17T08:43:05Z
---

# Issue #12: Tensors created and consumed inside a single subgraph cause ambiguous solutions

## Original post
- author: aheirman
- created_at: 2026-02-07T21:20:25Z
- url: https://github.com/yarongmu-google/MLSys/issues/12

<comment>
The statement:

>  Regarding the “tensors_to_retain” list, a list of lists where tensors_to_retain[k] specifies which output tensors (or loaded inputs) from Subgraph k should remain resident in the fast memory after the subgraph finishes. Any tensor not in this list is automatically evicted to the slow memory (if it is an output) or discarded (if it was an input).

Is not precise e.g.
Example 1B does not evict tensor 1 which is an output of Op0.
Example 3B does not evict tensor 1 which is an output of Op0.

If the rule should be interpreted as
> Any tensor not in this list is automatically evicted to the slow memory (if it is a used output of the subgraph) else it will be discarded.

There still exists situations where it has to evict a tensor in an unknown and unspecifiable subgroup:
```
graph LR
    Tensor0(("Tensor[0]<br>128x128"))
    Tensor1(("Tensor[1]<br>128x128"))
    Tensor2(("Tensor[2]<br>128x128"))
    Tensor3(("Tensor[3]<br>128x128"))
    Tensor4(("Tensor[4]<br>128x128"))
    Op0["Op[0]<br>1500"]
    Op1["Op[1]<br>1500"]
    Op2["Op[2]<br>1500"]
    Op3["Op[3]<br>1500"]

    Tensor0 --> Op0
    Op0 --> Tensor1
    Tensor1 --> Op1
    Tensor1 --> Op2
    Tensor1 --> Op3
    Op1 --> Tensor2
    Op2 --> Tensor3
    Op3 --> Tensor4
```

```
{
  "widths": [128,128,128,128,128],
  "heights": [128,128,128,128,128],
  "inputs": [[0],[1],[1], [1]],
  "outputs": [[1],[2],[3], [4]],
  "base_costs": [1500,1500,1500,1500],
  "op_types": ["Pointwise","Pointwise","Pointwise"],
  "fast_memory_capacity": 50000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}

{
  "subgraphs":[[0,1],[0,2],[3]],
  "granularities": [[128,128,1],[128,128,1],[128,128,1]],
  "tensors_to_retain": [[],[],[]],
  "traversal_orders": [null, null, null],
  "subgraph_latencies": [?,?,1638.4]
}
```

Which subgraph is required to evict tensor1 to give to subgraph2 ([3])?

Or should the rule be interpreted as:
> Any tensor not in this list is automatically evicted to the slow memory (if it is output of the last op of the subgraph) else it will be discarded.


</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-02-08T18:27:04Z
- url: https://github.com/yarongmu-google/MLSys/issues/12#issuecomment-3867823549

<comment>
Thanks for the question :)

The confusion likely stems from the distinction between ephemeral data (internal to a subgraph) and subgraph outputs.

- Ephemeral vs. output: While Tensor1 is indeed passed internally from Op0 to Op1 (ephemeral), it is also required by a future subgraph ([3]). This classifies it as an output of the subgraph.

- The eviction rule: The problem states that "Any tensor not in this list [tensors_to_retain] is automatically evicted to the slow memory (if it is an output)".

Since Tensor1 is an output for both [0, 1] and [0, 2], and strictly serialized execution applies, the system enforces an eviction at the end of both steps, unless you specify it in [tensors_to_retain].

In fact, for your example output, Tensor1 must be in [tensors_to_retain]; otherwise [3] can't run by ltself.
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-02-08T18:27:55Z
- url: https://github.com/yarongmu-google/MLSys/issues/12#issuecomment-3867827605

<comment>
I will close this for now. Please reopen if the above doesn't make sense :)
</comment>

---

## Comment 3
- author: aheirman
- created_at: 2026-02-09T22:24:39Z
- url: https://github.com/yarongmu-google/MLSys/issues/12#issuecomment-3874190324

<comment>
> In fact, for your example output, Tensor1 must be in [tensors_to_retain]; otherwise [3] can't run by ltself.

Indeed because of the latency.

So if I understand the valid (sometimes a "little inefficient") options are:

1) do not emit evict tensor 1, retain it in the first and second subgraph 
```
{
  "subgraphs":[[0,1],[0,2],[3]],
  "granularities": [[128,128,1],[128,128,1],[128,128,1]],
  "tensors_to_retain": [[1],[1],[]],
  "traversal_orders": [null, null, null],
  "subgraph_latencies": [3276.8,3276.8,1638.4]
}
```

2) evict tensor 1 in the first subgraph, retain in the second
```
{
  "subgraphs":[[0,1],[0,2],[3]],
  "granularities": [[128,128,1],[128,128,1],[128,128,1]],
  "tensors_to_retain": [[],[1],[]],
  "traversal_orders": [null, null, null],
  "subgraph_latencies": [4915.2,3276.8,1638.4]
}
```

3) evict tensor 1 in the second subgraph, retain in the first
```
{
  "subgraphs":[[0,1],[0,2],[3]],
  "granularities": [[128,128,1],[128,128,1],[128,128,1]],
  "tensors_to_retain": [[1],[],[]],
  "traversal_orders": [null, null, null],
  "subgraph_latencies": [3276.8,4915.2,3276.8]
}
```

4) evict tensor 1 in the first and second subgraph
```
{
  "subgraphs":[[0,1],[0,2],[3]],
  "granularities": [[128,128,1],[128,128,1],[128,128,1]],
  "tensors_to_retain": [[],[],[]],
  "traversal_orders": [null, null, null],
  "subgraph_latencies": [4915.2,4915.2,3276.8]
}
```

p.s. we can not re-open issues.
</comment>

---

## Comment 4
- author: xavierrouth
- created_at: 2026-02-10T23:49:00Z
- url: https://github.com/yarongmu-google/MLSys/issues/12#issuecomment-3881329359

<comment>
In the eviction rule: 

> The problem states that "Any tensor not in this list [tensors_to_retain] is automatically evicted to the slow memory (if it is an output)".

It Output meant to mean output of the entire graph, or output of the subgraph?

</comment>

---

