# CLAUDE.md

Context for Claude Code when working on this repository.

## What this repo is

An implementation of a memory-hierarchy-aware operator fusion and scheduling optimizer for the **MLSys 2026 Graph Scheduling Competition** (Track A: Systems, binary deliverable). The binary reads a DAG of tensor operations plus device parameters from JSON and writes an execution plan (subgraph partitioning + per-subgraph tile granularity + tensor retention + traversal order + latencies) to JSON. Target platform is Ubuntu 22.04 LTS, statically linked. The submission deadline is **April 24, 2026, 11:59 PM PT**.

Problem spec and clarifications live in an upstream repo, not here:
- `PROBLEM.md` — https://github.com/yarongmu-google/MLSys/blob/main/PROBLEM.md
- Issues — https://github.com/yarongmu-google/MLSys/issues (official clarifications are replies by `yarongmu-google`)

Treat replies by `yarongmu-google` as authoritative. Treat the spec as mutable — when something looks underspecified, the answer is usually in a numbered issue, not in inference.

## Binary interface

```
$ ./mlsys <path_to_input.json> <path_to_output.json>
```

Timeout is enforced externally by the harness. Just write the output file before getting killed. The contest also requires a `writeup.pdf` (2 pages) at submission time; that's not tracked here.

Benchmarks live in `benchmarks/mlsys-2026-*.json` (5 released, 20 withheld). Node counts in released set range from 5 (2s timeout) to 103 (60s). Hidden set goes up to 120s timeout, so graphs up to ~200 nodes are plausible.

## Module map (`src/`)

| Module | What it owns |
| :---- | :---- |
| `main.rs` | Binary entry. Currently a smoke test that prints the parsed input — **end-to-end plumbing is not yet wired up**. |
| `input_format.rs` | `InputFormat` and `DeviceParameters` serde-deserialized from the benchmark JSON. `DeviceParameters` is flattened into the input. |
| `graph.rs` | `ComputationGraph` (DAG), `Subgraph<'a>` (borrowing slice of nodes), `TopologicalOrder`, `TensorId`, `OperationId`, `OperationType`. The core data model. |
| `tiling.rs` | Tile-shape propagation (`propagate_tile_shape`), constraint unification (`ConstraintTracker`), tile search (`search_tile_values`). |
| `performance_model.rs` | `PerformanceMetric` (compute + memory, `Fraction`-exact), `subgraph_latency` (per-output, per-tile latency map). |
| `global_optimization.rs` | `extract_convex_subgraphs` (Korch-style pairwise enumeration) + `optimize_execution_plan` (BLP via `good_lp` + HiGHS). |
| `testutil.rs` | `#[cfg(test)]`-only helpers: `make_input`, `graph_from_edges`, `pointwise_graph`, `make_graph`, `subgraph`, `load_input`. |

Test fixtures: `tests/fixtures/official_example{1..5}.json`, one per PROBLEM.md example.

## Core types and invariants

- **`Subgraph<'a>.nodes` is kept sorted by topological position.** `from_nodes` enforces it; `insert` uses binary search to maintain it; `is_subset`, `subtract` rely on it for O(n+m) merge-style traversal. Don't push unsorted. Don't reorder.
- **`ComputationGraph.topological_order` is a `OnceCell`.** Cheap to call `topological_sort()` repeatedly.
- **`Fraction` is used throughout `performance_model.rs` for exact arithmetic.** Don't convert to `f64` until you have to (e.g. when passing costs to `good_lp`). Tests assert fractions like `Fraction::new(32768u64, 10u64)` — that's 3276.8 exactly.
- **`Axis` variants** are `TiledM`, `TiledN`, `TiledK`, `Full(i64)`. `ConstraintTracker` is a Union-Find over them with a "constant" tag (represented by finding a `Full(_)` parent). A MatMul constrains its LHS/RHS in the standard way; a Pointwise op equates both of its input axes to its output axes.
- **`PerformanceMetric::latency()` is `max(compute_cost, memory_cost)`** — this is the roofline. Sum `PerformanceMetric`s, don't sum latencies pre-maxed.

## What's implemented (and works, per tests)

- Full DAG construction from `InputFormat`, with cached topological sort.
- Subgraph set algebra: `is_subset`, `subtract`, `insert`, `contains`, `components`, `input_tensor_ids`, `output_tensor_ids`.
- Tile-shape propagation backward from a subgraph output through both MatMul and Pointwise ops, with Union-Find collapsing (`propagate_tile_shape`).
- Per-subgraph tile search (`search_tile_values`) that:
  - propagates each output's tile shape independently, then merges constraints,
  - iterates `m` and `n` candidates,
  - binary-searches the largest feasible `k` given the working-set limit minus retained-tensor reservation,
  - picks the `(m, n, k)` with minimum input traffic among feasible candidates.
- Per-subgraph latency (`subgraph_latency`) that:
  - iterates `(m_tile, n_tile, k_tile)` triples,
  - walks back from each subgraph output to find which input slices each tile needs (`input_tiles_for_output`),
  - tracks `cached_inputs` across adjacent tiles for **intra-subgraph implicit reuse** (this is the Issue #37/#65 behaviour),
  - scales per-op compute by `k_slice / K` (the Issue #10/#27 formula),
  - charges output eviction only on the last k-step of each spatial tile.
- Convex subgraph enumeration via execution-states DFS + pairwise subtract (`extract_convex_subgraphs`).
- BLP selection via `good_lp` (`optimize_execution_plan`) minimizing `Σ c_i · u_i` subject to:
  - producer constraints: `(Σ subgraphs producing tensor t as output) ≥ u_j` for every subgraph `j` consuming `t`,
  - output coverage: `(Σ subgraphs containing op) ≥ 1` for every op producing a graph output.
  - This encodes Korch's dependency constraint + a cover-based objective (supports recomputation naturally).

The BLP tests cover the key topologies, including ones where recomputation wins (`fanout_recompute`, `chain_recompute`, `wide_fanout_recompute`).

## What's missing / in flight

**Main gap: end-to-end integration.** `main.rs` still just prints the input. The pipeline needs to be wired:

```
parse InputFormat
  → extract_convex_subgraphs(graph)
  → for each subgraph:
      search_tile_values (skip if infeasible)
      → subgraph_latency → total cost
  → optimize_execution_plan(graph, costs)
  → for each selected subgraph: compute retained tensors, traversal order, per-subgraph latency
  → serialize output JSON (schema per PROBLEM.md § Output Format)
```

**Known algorithmic limitations:**

1. **`extract_convex_subgraphs` is the Korch pairwise-difference approach.** It's O(|execution_states|² × subtract-cost), which blows up on wide graphs. For benchmark 17 (103 nodes), this likely won't finish in 60s. A seed-and-grow connected-convex enumeration with canonical-rep dedup is the intended replacement — delay O(|V|+|E|) per candidate, no output-subset factor. Output tensors are deterministic in this problem (tensors with consumers outside the subgraph), so Korch's `2^|P'|` output-subset enumeration can be dropped entirely; the current code already does this structurally but the enumeration itself is still quadratic in execution-state count.
2. **Retention and traversal order are not yet chosen** — `subgraph_latency` accepts both but nothing upstream decides them. Retention: the Issue #34 one-step-lifetime constraint makes this a local, per-boundary greedy choice (retain if `2·tensor_size / bandwidth` saving exceeds granularity-downgrade cost from reduced fast-memory headroom in the succeeding subgraph). Traversal: for each subgraph, try raster and snake and keep whichever gives lower `subgraph_latency`.
3. **Fusion decisions don't yet consult compute-invariance**: total compute work for a subgraph at fixed spatial tile equals `Σ base_costs` regardless of `k`-split — `k` only moves latency through roofline tipping and accumulator-streaming traffic. Useful for pruning the k-search.
4. **No output serialization yet.** Format (from PROBLEM.md):

```json
{
  "subgraphs": [[0, 1], [2]],
  "granularities": [[64, 64, 128], [128, 128, 1]],
  "tensors_to_retain": [[1], []],
  "traversal_orders": [[0, 1, 3, 2], null],
  "subgraph_latencies": [2048.0, 1024.0]
}
```

## Issue-based clarifications to encode (upstream repo, numbered)

Treat these as tests-in-waiting:

- **#10, #27** — per-step compute when k-splitting is `base_cost × (k/K)` per op.
- **#28** — non-divisible `k` splits to slice sizes like `[43, 43, 42]` for `K=128, k=43`. Last step's compute scales proportionally to actual slice size (implementation currently does this; see `reduction_tile_size` in `performance_model.rs`).
- **#34** — retained tensors live for exactly one step. Don't try multi-hop retention.
- **#37** — raster order gets implicit reuse (not only snake). Example 4A latency is 7096, not 8192.
- **#59** — (a) partial-tensor reuse across different access patterns (same tensor as LHS in one op and RHS in another) is NOT permitted — separate partial copies; (b) Bélády-style arbitrary retention NOT permitted — only adjacent-tile reuse within a step; (c) eviction is whole-block only.
- **#65** — full-tensor reuse within a single fused subgraph IS permitted: if T1 feeds multiple ops in the same subgraph, T1 loads once.
- **Open (filed, awaiting reply)** — output tensor computation order within a fused subgraph when multiple outputs share an input accessed in different block shapes. Ordering affects implicit reuse at phase boundaries. Until answered, don't rely on either ordering being canonical.

## Reference numbers for sanity-checking

From PROBLEM.md examples (all already covered by `performance_model.rs` tests):

- Example 1 Strategy A (two separate pointwise subgraphs): 3276.8 + 3276.8 = 6553.6.
- Example 1 Strategy B (merged at [128,128,1]): 3276.8.
- Example 1 Strategy C (merged at [64,64,1], compute-bound): 4400.
- Example 3 Strategy A (diamond spill): 11468.8 = 3276.8 + 3276.8 + 4915.2.
- Example 3 Strategy B ("Flash" recomputation, Op0 duplicated): 6276.8.
- Example 3 Strategy C (selective residency, retain T1): 4638.4.
- Example 4 raster at (64,64,128): 7096.
- Example 5 fused at [128,128,32] (capacity 45000, native [128,128]): 6915.2. k=64 and k=128 both OOM.

## Coding conventions observed

- Snake_case everywhere. No `anyhow`/`thiserror` — panics in `main`, `Result` with a local enum in `tiling::SearchError`. Keep that style unless a function is fallible in a genuinely complex way.
- No `unsafe`. No `unwrap` on `Option` except where topological invariants make it unreachable (e.g. after an `is_empty` guard).
- Tests document the graph with ASCII art comments before the `#[test]`. Match that style — the fixtures are visual and it makes diffs legible. See any test in `graph.rs`, `tiling.rs`, `performance_model.rs`.
- Integer widths: `i64` throughout for dimensions and costs, `usize` for IDs wrapped in newtypes (`TensorId(pub usize)`, `OperationId(pub usize)`).
- `HashSet`/`HashMap` from `std::collections` — no third-party hashers.
- `Fraction` for costs in `performance_model.rs`. Do not mix with `f64` except at the BLP boundary (converting `Fraction` → `f64` for `good_lp` costs).

## Testing conventions

- Unit tests live inline (`#[cfg(test)] mod tests { ... }`) in each module.
- Fixtures under `tests/fixtures/` loaded via `testutil::load_input(filename)` which uses `CARGO_MANIFEST_DIR`.
- Each PROBLEM.md strategy has at least one named test (`example3_strategy_c_selective_residency`, etc.). When adding features that affect a strategy's cost, add/update the corresponding test first.
- `graph_from_edges` is the shorthand for synthetic graphs in `global_optimization` tests; the op and tensor ids follow a fixed convention (op `i` produces tensor `i`, external inputs use high-numbered tensors).

## What NOT to do

- **Don't adopt FFM or AccelForge.** Considered and explicitly abandoned. The one-Einsum-per-compute-node invariant blocks the recomputation patterns this problem rewards.
- **Don't enumerate output subsets of a candidate subgraph as independent candidates.** Korch does this (`2^|P'|` factor) because some kernel outputs are discardable. In this problem the output set is *determined* by the subgraph (tensors with consumers outside). One convex subgraph → one candidate.
- **Don't assume divisibility of `k` into `K`.** Non-divisible splits are legal per Issue #28; the last step has a smaller slice.
- **Don't assume the default raster order is the only one considered.** Snake order reliably reduces memory traffic on 2×N and larger grids (Issue #37 confirms raster also reuses; snake reuses strictly more).

## Useful build / test commands

```bash
cargo build --release           # the submission binary
cargo test                      # all tests
cargo test performance_model    # one module
cargo test example5_strategy_b  # one test
cargo run --release -- benchmarks/mlsys-2026-1.json /tmp/out.json
```

## Key papers (in project knowledge, not the repo)

- **Korch** (ASPLOS 2024) — BLP formulation in `optimize_execution_plan` follows this; the code from `humuyan/Korch` confirms the tensor-level reading of eq. 4 (the paper's index notation conflates kernel and primitive indices).
- **Optimus** (LCTES 2021 / TECS 2022) — DP-based fusion on DAGs; relevant if we want to extend beyond BLP.
- **FlashTensor** (PPoPP 2025) — non-convex kernel mapping via tensor property analysis. The AI threshold (200 flops/element in their code) is A100-specific; don't port the number.
- **Welder** (OSDI 2023) — tile propagation via inter-layer independence. The backward tile-shape propagation in `tiling.rs` is conceptually aligned.
- **He & Yu** (MLSys 2023) — fusion-aware min-cut rematerialization. Doesn't directly apply (training focus, bans compute-bound ops from recomputation), but the min-cut framing is useful context.
- **ROLLER** (OSDI 2022) — rTile alignment with native granularity; justifies snapping spatial dims to multiples of `native_granularity`.
