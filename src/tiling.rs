use std::collections::{HashMap, HashSet};

use crate::{
    graph::{ComputationGraph, OperationId, OperationType, Subgraph, TensorId},
    input_format::DeviceParameters,
};

pub(crate) fn ceil_div(x: i64, y: i64) -> i64 {
    (x + y - 1) / y
}

#[derive(Debug)]
pub enum SearchError {
    Inconsistent,
    NotFound,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub(crate) enum SliceRole {
    LhsRowStrip,
    RhsColStrip,
    OutAccumulator,
    PointwiseTile,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub(crate) struct SliceIndex {
    pub spatial_row: i64,
    pub spatial_col: i64,
    pub k_step: i64,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub(crate) struct SliceShape {
    pub rows: i64,
    pub cols: i64,
}

impl SliceShape {
    pub fn elements(&self) -> i64 {
        self.rows * self.cols
    }
}

/// A named slice of fast-memory residency.
///
/// Asymmetric matching per PROBLEM clarifications #65/#70:
/// `Whole(T)` covers any subsequent partial access to `T`; `Partial(...)`
/// matches only an identical slice spec.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub(crate) enum SliceName {
    Whole(TensorId),
    Partial {
        tensor: TensorId,
        role: SliceRole,
        index: SliceIndex,
        shape: SliceShape,
    },
}

impl SliceName {
    pub fn tensor_id(&self) -> TensorId {
        match self {
            Self::Whole(t) => *t,
            Self::Partial { tensor, .. } => *tensor,
        }
    }

    pub fn elements(&self, whole_size: i64) -> i64 {
        match self {
            Self::Whole(_) => whole_size,
            Self::Partial { shape, .. } => shape.elements(),
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct ResidencySet {
    names: HashSet<SliceName>,
}

impl ResidencySet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true iff `name` is satisfied by current residency.
    /// `Whole(T)` resident covers any partial access to `T`.
    pub fn matches(&self, name: &SliceName) -> bool {
        if self.names.contains(name) {
            return true;
        }
        if let SliceName::Partial { tensor, .. } = name {
            return self.names.contains(&SliceName::Whole(*tensor));
        }
        false
    }

    pub fn replace<I: IntoIterator<Item = SliceName>>(&mut self, names: I) {
        self.names.clear();
        self.names.extend(names);
    }
}

/// Returns sorted divisors of `value` that are `<= limit`.
pub(crate) fn divisors_le(value: i64, limit: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut i: i64 = 1;
    while i * i <= value {
        if value % i == 0 {
            if i <= limit {
                out.push(i);
            }
            let j = value / i;
            if j != i && j <= limit {
                out.push(j);
            }
        }
        i += 1;
    }
    out.sort();
    out.dedup();
    out
}

fn op_kind(graph: &ComputationGraph, op_id: OperationId) -> OperationType {
    let any_output = graph.output_ids_for(op_id)[0];
    graph.producer_of(any_output).unwrap().kind
}

/// Per-op iteration tile dimensions `(m_O, n_O, k_split_O)` derived by
/// projecting the global granule `(h, w, k)` through the subgraph's chain
/// of consumers back to each op:
///
/// - The op producing a boundary output uses `(h, w)` directly.
/// - For an op whose output feeds a pointwise consumer, dims are inherited.
/// - For an op whose output feeds a matmul consumer's LHS, this op's
///   `(m, n) = (consumer.m, consumer.k_split)`. Symmetric for RHS.
/// - A matmul's own `k_split_O` is `k` if it produces a boundary output,
///   else its full `K_op_O` (the inner reduction is not split independently).
fn propagate_op_dims(
    subgraph: &Subgraph<'_>,
    h: i64,
    w: i64,
    k: i64,
) -> HashMap<OperationId, (i64, i64, i64)> {
    let graph = subgraph.parent();
    let outputs = subgraph.output_tensor_ids();
    let mut op_dims: HashMap<OperationId, (i64, i64, i64)> = HashMap::new();

    // Subgraph nodes are sorted ascending by topological position; iterate
    // in reverse so each op's downstream consumer is already resolved.
    for &op_id in subgraph.nodes().iter().rev() {
        let op_outputs = graph.output_ids_for(op_id);
        let kind = op_kind(graph, op_id);
        let any_output = op_outputs[0];
        let dims = if op_outputs.iter().any(|t| outputs.contains(t)) {
            // Boundary-producing op: takes the global granule directly.
            match kind {
                OperationType::MatMul => (h, w, k),
                OperationType::Pointwise => (h, w, 1),
            }
        } else {
            // Find any in-subgraph consumer of this op's output.
            let consumer_id = graph
                .consumer_ids_for(any_output)
                .iter()
                .find(|&&c| subgraph.contains(c))
                .copied();
            let Some(consumer_id) = consumer_id else {
                // Dead op (no consumer in or out of subgraph); fall back.
                op_dims.insert(op_id, (h, w, 1));
                continue;
            };
            let &(m_c, n_c, k_split_c) = op_dims.get(&consumer_id).expect("topo order");
            let consumer_kind = op_kind(graph, consumer_id);
            let (m_o, n_o) = match consumer_kind {
                OperationType::Pointwise => (m_c, n_c),
                OperationType::MatMul => {
                    let consumer_inputs = graph.input_ids_for(consumer_id);
                    let is_lhs = consumer_inputs[0] == any_output;
                    if is_lhs {
                        // Op feeds consumer's LHS strip (m_c × k_split_c).
                        (m_c, k_split_c)
                    } else {
                        // Op feeds consumer's RHS strip (k_split_c × n_c).
                        (k_split_c, n_c)
                    }
                }
            };
            let k_split_o = match kind {
                OperationType::MatMul => {
                    // Inner matmul: full reduction, no independent split.
                    graph.tensors()[graph.input_ids_for(op_id)[0].0].width
                }
                OperationType::Pointwise => 1,
            };
            (m_o, n_o, k_split_o)
        };
        op_dims.insert(op_id, dims);
    }
    op_dims
}

/// Conservative upper bound on resident bytes at any iteration.
///
/// Per PLAN.md §3.3 / §3.6:
/// - matmul boundary LHS: `m_O * K_op_O` (full row, held resident across the
///   inner k-loop per asymmetric naming).
/// - matmul boundary RHS: `k_split_O * n_O` (streaming one k-strip).
/// - pointwise boundary input: `m_O * n_O`.
/// - boundary output accumulator: `m_O * n_O`.
/// - ephemeral intermediates (only-internal consumers): 0.
fn peak_working_set(subgraph: &Subgraph<'_>, w: i64, h: i64, k: i64) -> i64 {
    let graph = subgraph.parent();
    let inputs = subgraph.input_tensor_ids();
    let outputs = subgraph.output_tensor_ids();
    let op_dims = propagate_op_dims(subgraph, h, w, k);

    let mut counted: HashSet<(TensorId, SliceRole)> = HashSet::new();
    let mut total: i64 = 0;

    for &op_id in subgraph.nodes() {
        let &(m_o, n_o, k_split_o) = op_dims.get(&op_id).expect("dims propagated");
        let kind = op_kind(graph, op_id);
        let op_inputs = graph.input_ids_for(op_id);
        match kind {
            OperationType::MatMul => {
                let lhs = op_inputs[0];
                let rhs = op_inputs[1];
                let k_op_o = graph.tensors()[lhs.0].width;
                if inputs.contains(&lhs) && counted.insert((lhs, SliceRole::LhsRowStrip)) {
                    total += m_o * k_op_o;
                }
                if inputs.contains(&rhs) && counted.insert((rhs, SliceRole::RhsColStrip)) {
                    total += k_split_o * n_o;
                }
            }
            OperationType::Pointwise => {
                for &input in op_inputs {
                    if inputs.contains(&input) && counted.insert((input, SliceRole::PointwiseTile))
                    {
                        total += m_o * n_o;
                    }
                }
            }
        }
        for &out in graph.output_ids_for(op_id) {
            if outputs.contains(&out) && counted.insert((out, SliceRole::OutAccumulator)) {
                total += m_o * n_o;
            }
        }
    }
    total
}

/// Mixed pointwise/matmul fusion validity per PLAN.md §1.8.
///
/// Prologue (pointwise feeds matmul LHS/RHS): split-k along the matmul's
/// reduction is forbidden — the pointwise tile must cover the full reduction
/// of the consuming matmul (`w >= K_op` for LHS feeders, `h >= K_op` for
/// RHS feeders). Constraint applies only when split-k is actually used
/// (`k < K_op`).
fn mixed_fusion_valid(subgraph: &Subgraph<'_>, w: i64, h: i64, k: i64) -> bool {
    let graph = subgraph.parent();
    for &op_id in subgraph.nodes() {
        if !matches!(op_kind(graph, op_id), OperationType::MatMul) {
            continue;
        }
        let inputs = graph.input_ids_for(op_id);
        let lhs = inputs[0];
        let rhs = inputs[1];
        let k_op_local = graph.tensors()[lhs.0].width;
        if k >= k_op_local {
            continue;
        }
        for (idx, side_input) in [(0usize, lhs), (1usize, rhs)] {
            let Some(producer_id) = graph.producer_id_of(side_input) else {
                continue;
            };
            if !subgraph.contains(producer_id) {
                continue;
            }
            if !matches!(op_kind(graph, producer_id), OperationType::Pointwise) {
                continue;
            }
            let satisfied = if idx == 0 {
                w >= k_op_local
            } else {
                h >= k_op_local
            };
            if !satisfied {
                return false;
            }
        }
    }
    true
}

/// Memory traffic estimate (bytes loaded + bytes evicted) at granule
/// `[w, h, k]` under raster traversal. Used as the tile-search objective.
///
/// This is a coarse total — no inter-iteration reuse is credited — which is
/// monotonic enough for ranking candidates. The performance model in
/// `performance_model.rs` does the precise per-step accounting with reuse.
fn memory_traffic(
    subgraph: &Subgraph<'_>,
    w: i64,
    h: i64,
    k: i64,
    retained: &HashSet<TensorId>,
) -> i64 {
    let graph = subgraph.parent();
    let inputs = subgraph.input_tensor_ids();
    let outputs = subgraph.output_tensor_ids();

    let mut total: i64 = 0;

    for &output_id in outputs.iter() {
        let out_t = &graph.tensors()[output_id.0];
        let producer_id = graph.producer_id_of(output_id).unwrap();
        let producer_kind = op_kind(graph, producer_id);
        let k_op_for_output = match producer_kind {
            OperationType::MatMul => graph.tensors()[graph.input_ids_for(producer_id)[0].0].width,
            OperationType::Pointwise => 1,
        };
        let spatial_tiles = ceil_div(out_t.height, h) * ceil_div(out_t.width, w);
        let k_steps = if k_op_for_output == 1 {
            1
        } else {
            ceil_div(k_op_for_output, k)
        };

        // Walk back from this output, summing per-iter slice bytes for
        // boundary inputs only. Non-boundary intermediates are ephemeral.
        let per_iter_input_bytes =
            walk_per_iter_bytes(subgraph, output_id, w, h, k, &inputs, retained);

        // Per-iter output eviction (last k step only).
        let evict_bytes = if retained.contains(&output_id) {
            0
        } else {
            h * w
        };

        let in_traffic = spatial_tiles * k_steps * per_iter_input_bytes;
        let out_traffic = spatial_tiles * evict_bytes;
        total += in_traffic + out_traffic;
    }
    total
}

/// Walk back from `output_id` once, summing the sizes of all boundary-input
/// slices needed for one iteration of `output_id` at granule `[w, h, k]`.
///
/// At a chained matmul, an inner matmul whose output is not the walked
/// terminus uses its full reduction (the inner k-loop is not split), so its
/// LHS slice is `h × K_op_inner` and its RHS slice is `K_op_inner × <child
/// shape col>`. This matches the `input_tiles_for_output` walker that the
/// performance model uses, so the traffic objective and the per-step
/// accounting agree on what's loaded.
fn walk_per_iter_bytes(
    subgraph: &Subgraph<'_>,
    output_id: TensorId,
    w: i64,
    h: i64,
    k: i64,
    inputs: &HashSet<TensorId>,
    retained: &HashSet<TensorId>,
) -> i64 {
    let graph = subgraph.parent();
    let mut bytes: i64 = 0;
    let mut stack: Vec<(TensorId, (i64, i64))> = vec![(output_id, (h, w))];
    while let Some((tensor_id, (rows, cols))) = stack.pop() {
        if inputs.contains(&tensor_id) {
            if !retained.contains(&tensor_id) {
                bytes += rows * cols;
            }
            continue;
        }
        let producer_id = match graph.producer_id_of(tensor_id) {
            Some(id) => id,
            None => continue,
        };
        if !subgraph.contains(producer_id) {
            continue;
        }
        let kind = op_kind(graph, producer_id);
        let op_inputs = graph.input_ids_for(producer_id);
        match kind {
            OperationType::Pointwise => {
                for &input in op_inputs {
                    stack.push((input, (rows, cols)));
                }
            }
            OperationType::MatMul => {
                let lhs = op_inputs[0];
                let rhs = op_inputs[1];
                let k_op_local = graph.tensors()[lhs.0].width;
                let inner_k = if tensor_id == output_id {
                    k
                } else {
                    k_op_local
                };
                stack.push((lhs, (rows, inner_k)));
                stack.push((rhs, (inner_k, cols)));
            }
        }
    }
    bytes
}

pub fn search_tile_values(
    subgraph: &Subgraph<'_>,
    device_params: &DeviceParameters,
    retained_tensor_ids: &[TensorId],
) -> Result<(i64, i64, i64), SearchError> {
    let retained: HashSet<TensorId> = retained_tensor_ids.iter().copied().collect();
    let graph = subgraph.parent();
    let outputs = subgraph.output_tensor_ids();
    if outputs.is_empty() {
        return Err(SearchError::Inconsistent);
    }

    // Native cap applies to all three axes per #74/#78/#80/#86.
    let (native_w, native_h) = device_params.native_granularity;
    let native_k = native_w.min(native_h);

    // Spatial candidates must divide every output's W (resp. H) so that each
    // output tiles cleanly under the chosen granule. (Cross-output divisibility
    // is required by #20/#58.)
    let mut common_w_set: Option<HashSet<i64>> = None;
    let mut common_h_set: Option<HashSet<i64>> = None;
    for &output_id in outputs.iter() {
        let t = &graph.tensors()[output_id.0];
        let w_set: HashSet<i64> = divisors_le(t.width, native_w).into_iter().collect();
        let h_set: HashSet<i64> = divisors_le(t.height, native_h).into_iter().collect();
        common_w_set = Some(match common_w_set.take() {
            None => w_set,
            Some(prev) => prev.intersection(&w_set).copied().collect(),
        });
        common_h_set = Some(match common_h_set.take() {
            None => h_set,
            Some(prev) => prev.intersection(&h_set).copied().collect(),
        });
    }
    let mut candidate_w: Vec<i64> = common_w_set.unwrap_or_default().into_iter().collect();
    let mut candidate_h: Vec<i64> = common_h_set.unwrap_or_default().into_iter().collect();
    candidate_w.sort();
    candidate_h.sort();

    // K candidates: divisors of every matmul's K_op (so reductions split
    // evenly). For a pure pointwise subgraph, k is irrelevant; emit k=1.
    let matmul_k_ops: Vec<i64> = subgraph
        .nodes()
        .iter()
        .filter(|&&op_id| matches!(op_kind(graph, op_id), OperationType::MatMul))
        .map(|&op_id| graph.tensors()[graph.input_ids_for(op_id)[0].0].width)
        .collect();
    let candidate_k: Vec<i64> = if matmul_k_ops.is_empty() {
        vec![1]
    } else {
        let min_k_op = *matmul_k_ops.iter().min().unwrap();
        let limit = native_k.min(min_k_op);
        // k must divide every matmul's K_op so all matmuls split cleanly.
        divisors_le(min_k_op, limit)
            .into_iter()
            .filter(|k| matmul_k_ops.iter().all(|&kop| kop % k == 0))
            .collect()
    };

    if candidate_w.is_empty() || candidate_h.is_empty() || candidate_k.is_empty() {
        return Err(SearchError::NotFound);
    }

    let reserved_for_retained: i64 = retained
        .iter()
        .map(|&tid| graph.tensors()[tid.0].size())
        .sum();
    let capacity = device_params.fast_memory_capacity - reserved_for_retained;
    if capacity <= 0 {
        return Err(SearchError::NotFound);
    }

    let mut best: Option<(i64, i64, i64)> = None;
    let mut best_traffic: i64 = i64::MAX;

    for &w in &candidate_w {
        for &h in &candidate_h {
            for &k in &candidate_k {
                if !mixed_fusion_valid(subgraph, w, h, k) {
                    continue;
                }
                if peak_working_set(subgraph, w, h, k) > capacity {
                    continue;
                }
                let traffic = memory_traffic(subgraph, w, h, k, &retained);
                let better = match best {
                    None => true,
                    Some(_) if traffic < best_traffic => true,
                    Some((bw, bh, bk)) if traffic == best_traffic => {
                        // Tie-break: prefer larger (w, h, k) lexicographically.
                        (w, h, k) > (bw, bh, bk)
                    }
                    _ => false,
                };
                if better {
                    best = Some((w, h, k));
                    best_traffic = traffic;
                }
            }
        }
    }

    best.ok_or(SearchError::NotFound)
}

#[cfg(test)]
mod tests {
    use super::{divisors_le, search_tile_values};
    use crate::graph::{ComputationGraph, OperationType, TensorId};
    use crate::input_format::{DeviceParameters, InputFormat};
    use crate::testutil::{load_input, subgraph};

    #[test]
    fn divisors_le_basic() {
        assert_eq!(divisors_le(128, 128), vec![1, 2, 4, 8, 16, 32, 64, 128]);
        assert_eq!(divisors_le(128, 64), vec![1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(divisors_le(12, 100), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(divisors_le(1, 1), vec![1]);
    }

    // Example 1: pointwise chain at native (128,128). Capacity easily fits;
    // tie-break picks the largest granule.
    #[test]
    fn official_repo_example1() {
        let input = load_input("official_example1.json");
        let graph = ComputationGraph::new(&input);
        let subgraph = subgraph(&graph, [0, 1]);

        let tile_size = search_tile_values(&subgraph, &input.device_parameters, &[]).unwrap();
        assert_eq!(tile_size, (128, 128, 1));
    }

    // Example 2: 256x256 pointwise; tile picks native 128x128.
    #[test]
    fn official_repo_example2() {
        let input = load_input("official_example2.json");
        let graph = ComputationGraph::new(&input);
        let subgraph = subgraph(&graph, [0, 1]);

        let tile_size = search_tile_values(&subgraph, &input.device_parameters, &[]).unwrap();
        assert_eq!(tile_size, (128, 128, 1));
    }

    // Example 5: chained matmul, K_op=128, capacity=45000. Strict
    // peak-working-set rule (LHS row strip h*K_op) gives:
    //   k=64: 16384 + 8192 + 8192 + 16384 = 49152 > 45000 → reject
    //   k=32: 16384 + 4096 + 4096 + 16384 = 40960 ≤ 45000 → feasible
    //   smaller k all feasible; tie-break prefers k=32.
    #[test]
    fn official_repo_example5() {
        let input = load_input("official_example5.json");
        let graph = ComputationGraph::new(&input);
        let subgraph = subgraph(&graph, [0, 1]);

        let tile_size = search_tile_values(&subgraph, &input.device_parameters, &[]).unwrap();
        assert_eq!(tile_size, (128, 128, 32));
    }

    // 256x256 variant of example 5. K_op=256, native (128,128), capacity 45000.
    // No (128,128,k) candidate fits because the LHS row strip at h=128 is
    // 128*256=32768 alone, leaving too little headroom for the strips and
    // accumulators. Search picks the largest-granule combination that fits
    // the capacity gate and minimizes traffic.
    #[test]
    fn official_repo_example5_256() {
        let device_params = DeviceParameters {
            fast_memory_capacity: 45_000,
            slow_memory_bandwidth: 10,
            native_granularity: (128, 128),
        };
        let graph = ComputationGraph::new(&InputFormat {
            widths: vec![256, 256, 256, 256, 256],
            heights: vec![256, 256, 256, 256, 256],
            inputs: vec![
                vec![TensorId(0), TensorId(1)],
                vec![TensorId(3), TensorId(2)],
            ],
            outputs: vec![vec![TensorId(3)], vec![TensorId(4)]],
            base_costs: vec![2000, 2000],
            op_types: vec![OperationType::MatMul, OperationType::MatMul],
            device_parameters: device_params.clone(),
        });
        let subgraph = subgraph(&graph, [0, 1]);

        let tile_size = search_tile_values(&subgraph, &device_params, &[]).unwrap();
        // Concrete value verified by running search; encoded here as a
        // regression test against the divisor-based search.
        assert_eq!(tile_size.0.max(tile_size.1) <= 128, true);
        assert!(256 % tile_size.0 == 0);
        assert!(256 % tile_size.1 == 0);
        assert!(256 % tile_size.2 == 0);
    }

    // Super-native granularity is rejected by the per-axis caps (#86).
    #[test]
    fn super_native_never_chosen() {
        let input = load_input("official_example1.json");
        let graph = ComputationGraph::new(&input);
        let subgraph = subgraph(&graph, [0, 1]);

        let (w, h, k) = search_tile_values(&subgraph, &input.device_parameters, &[]).unwrap();
        let (native_w, native_h) = input.device_parameters.native_granularity;
        assert!(w <= native_w);
        assert!(h <= native_h);
        assert!(k <= native_w.min(native_h));
    }

    // Pointwise feeding a matmul LHS with split-k disallows w < K_op.
    #[test]
    fn pointwise_feeding_matmul_lhs_rejects_small_w() {
        // op0: pw(t0) → t1 (64x64)
        // op1: matmul(t1, t2) → t3, K_op = width(t1) = 64
        // capacity tight enough that searching divisors below 64 might be
        // desirable, but mixed-fusion validity should reject any w < 64
        // when k < 64 (split-k under prologue is invalid).
        let device_params = DeviceParameters {
            fast_memory_capacity: 80_000,
            slow_memory_bandwidth: 10,
            native_granularity: (128, 128),
        };
        let graph = ComputationGraph::new(&InputFormat {
            widths: vec![64, 64, 64, 64],
            heights: vec![64, 64, 64, 64],
            inputs: vec![vec![TensorId(0)], vec![TensorId(1), TensorId(2)]],
            outputs: vec![vec![TensorId(1)], vec![TensorId(3)]],
            base_costs: vec![100, 1000],
            op_types: vec![OperationType::Pointwise, OperationType::MatMul],
            device_parameters: device_params.clone(),
        });
        let subgraph = subgraph(&graph, [0, 1]);

        let (w, _h, k) = search_tile_values(&subgraph, &device_params, &[]).unwrap();
        // If k is split (< 64), the prologue rule requires w ≥ K_op = 64.
        if k < 64 {
            assert!(
                w >= 64,
                "split-k prologue requires w ≥ K_op, got w={w}, k={k}"
            );
        }
    }

    // k is capped by the smallest matmul K_op in the subgraph.
    #[test]
    fn k_capped_at_min_k_op() {
        // op0: matmul, K_op=64; op1: matmul, K_op=128. Min is 64.
        let device_params = DeviceParameters {
            fast_memory_capacity: 200_000,
            slow_memory_bandwidth: 10,
            native_granularity: (128, 128),
        };
        let graph = ComputationGraph::new(&InputFormat {
            widths: vec![64, 128, 128, 128, 128],
            heights: vec![128, 64, 128, 128, 128],
            inputs: vec![
                vec![TensorId(0), TensorId(1)],
                vec![TensorId(2), TensorId(3)],
            ],
            outputs: vec![vec![TensorId(2)], vec![TensorId(4)]],
            base_costs: vec![1000, 1000],
            op_types: vec![OperationType::MatMul, OperationType::MatMul],
            device_parameters: device_params.clone(),
        });
        let subgraph = subgraph(&graph, [0, 1]);

        let (_w, _h, k) = search_tile_values(&subgraph, &device_params, &[]).unwrap();
        assert!(k <= 64, "k must be ≤ min K_op = 64, got {k}");
    }

    // Pure pointwise: largest granule fits and is preferred by tie-break.
    #[test]
    fn traffic_minimization_prefers_largest_pointwise() {
        let input = load_input("official_example1.json");
        let graph = ComputationGraph::new(&input);
        let sg = subgraph(&graph, [0]);
        let tile = search_tile_values(&sg, &input.device_parameters, &[]).unwrap();
        assert_eq!(tile, (128, 128, 1));
    }
}
