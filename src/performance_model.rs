use std::{
    cmp::{self, Ordering},
    collections::{HashMap, HashSet},
    ops::{Add, AddAssign},
};

use fraction::Fraction;

use crate::{
    graph::{OperationType, Subgraph, TensorId},
    input_format::DeviceParameters,
    tiling::{ResidencySet, SliceIndex, SliceName, SliceRole, SliceShape, ceil_div},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerformanceMetric {
    compute_cost: Fraction,
    memory_cost: Fraction,
}

impl PerformanceMetric {
    pub fn zero() -> Self {
        Self {
            compute_cost: Fraction::from(0i64),
            memory_cost: Fraction::from(0i64),
        }
    }

    pub fn from_compute_cost(cost: Fraction) -> Self {
        Self {
            compute_cost: cost,
            memory_cost: Fraction::from(0i64),
        }
    }

    pub fn from_memory_cost(cost: Fraction) -> Self {
        Self {
            compute_cost: Fraction::from(0i64),
            memory_cost: cost,
        }
    }

    pub fn latency(&self) -> Fraction {
        cmp::max(self.memory_cost, self.compute_cost)
    }
}

impl PartialOrd for PerformanceMetric {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PerformanceMetric {
    fn cmp(&self, other: &Self) -> Ordering {
        self.latency().cmp(&other.latency())
    }
}

impl Add for PerformanceMetric {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            compute_cost: self.compute_cost + rhs.compute_cost,
            memory_cost: self.memory_cost + rhs.memory_cost,
        }
    }
}

impl AddAssign for PerformanceMetric {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Walks back from `output_id` and returns the list of slice names that must
/// be resident for one iteration `(m_idx, n_idx, k_idx)` at granule
/// `tile_size`.
///
/// The walk crosses fused matmul/pointwise ops:
/// - Through a matmul whose output is the iteration's terminus: split-k along
///   that op's reduction is honored (`reduction_size = tile_k`).
/// - Through an inner matmul: full reduction (its own k-axis is not split).
/// - Through a pointwise: shape is preserved.
///
/// The returned slice names use:
/// - `LhsRowStrip` / `RhsColStrip` for matmul boundary inputs;
/// - `PointwiseTile` for pointwise boundary inputs (also used as the
///   default for the walked-from output's slice if it's a boundary input
///   directly — this can't happen in practice but we have to pick a role).
fn slice_names_for_output(
    subgraph: &Subgraph<'_>,
    output_id: TensorId,
    tile_size: (i64, i64, i64),
    tile_index: (i64, i64, i64),
) -> Vec<SliceName> {
    let input_tensor_ids = subgraph.input_tensor_ids();
    let graph = subgraph.parent();
    let output_tensor = &graph.tensors()[output_id.0];
    let (tile_h, tile_w, tile_k) = tile_size;
    let (m_idx, n_idx, k_idx) = tile_index;

    let output_position = (m_idx * tile_h, n_idx * tile_w);
    let output_shape = (
        i64::min(tile_h, output_tensor.height - output_position.0),
        i64::min(tile_w, output_tensor.width - output_position.1),
    );

    // Walk-stack entries: (tensor, position, shape, role).
    let mut stack: Vec<(TensorId, (i64, i64), (i64, i64), SliceRole)> = vec![(
        output_id,
        output_position,
        output_shape,
        SliceRole::PointwiseTile,
    )];
    let mut result: Vec<SliceName> = Vec::new();

    while let Some((tensor_id, position, shape, role)) = stack.pop() {
        if input_tensor_ids.contains(&tensor_id) {
            result.push(SliceName::Partial {
                tensor: tensor_id,
                role,
                index: SliceIndex {
                    spatial_row: position.0,
                    spatial_col: position.1,
                    k_step: 0,
                },
                shape: SliceShape {
                    rows: shape.0,
                    cols: shape.1,
                },
            });
            continue;
        }

        let producer_id = graph.producer_id_of(tensor_id).unwrap();
        let producer_op = graph.producer_of(tensor_id).unwrap();
        let input_ids = graph.input_ids_for(producer_id);
        match producer_op.kind {
            OperationType::Pointwise => {
                for &input_id in input_ids {
                    stack.push((input_id, position, shape, SliceRole::PointwiseTile));
                }
            }
            OperationType::MatMul => {
                let input0_id = input_ids[0];
                let input0 = &graph.tensors()[input0_id.0];
                let input1_id = input_ids[1];

                let reduction_index = if tensor_id == output_id {
                    k_idx * tile_k
                } else {
                    0
                };
                let reduction_size = if tensor_id == output_id {
                    i64::min(tile_k, input0.width - reduction_index)
                } else {
                    input0.width
                };
                stack.push((
                    input0_id,
                    (position.0, reduction_index),
                    (shape.0, reduction_size),
                    SliceRole::LhsRowStrip,
                ));
                stack.push((
                    input1_id,
                    (reduction_index, position.1),
                    (reduction_size, shape.1),
                    SliceRole::RhsColStrip,
                ));
            }
        }
    }
    result
}

pub fn subgraph_latency(
    device_params: &DeviceParameters,
    subgraph: &Subgraph<'_>,
    tile_size: (i64, i64, i64),
    retained_tensor_ids: &[TensorId],
) -> HashMap<TensorId, Vec<PerformanceMetric>> {
    let retained_tensor_ids: HashSet<TensorId> = retained_tensor_ids.iter().copied().collect();
    let input_tensor_ids = subgraph.input_tensor_ids();
    let output_tensor_ids: Vec<TensorId> = {
        let mut v: Vec<TensorId> = subgraph.output_tensor_ids().into_iter().collect();
        v.sort();
        v
    };

    let graph = subgraph.parent();
    let (tile_h, tile_w, tile_k) = tile_size;

    let tile_counts = {
        let output_id = output_tensor_ids
            .iter()
            .next()
            .expect("subgraph must have at least one output");
        let output = &graph.tensors()[output_id.0];
        (
            ceil_div(output.height, tile_h),
            ceil_div(output.width, tile_w),
        )
    };

    let bandwidth = device_params.slow_memory_bandwidth;

    // Adjacent-only implicit reuse (PROBLEM #59 / #65 / #70): a slice loaded
    // at iteration i is available at i+1 iff the same name is in
    // `cached_inputs`. Asymmetric matching: a `Whole(T)` resident covers any
    // partial access to T — currently we never promote to Whole, but the
    // matcher honors the rule for future use.
    let input_memory_cost = |names: &[SliceName], cached: &ResidencySet| -> Fraction {
        let bytes: i64 = names
            .iter()
            .map(|name| {
                let tid = name.tensor_id();
                if retained_tensor_ids.contains(&tid) || cached.matches(name) {
                    0
                } else {
                    let whole_size = graph.tensors()[tid.0].size();
                    name.elements(whole_size)
                }
            })
            .sum();
        Fraction::from(bytes) / bandwidth
    };

    // Total per-spatial-tile compute charge for ops feeding `output_id`.
    // For chained subgraphs, this sums every contributing op's `base_cost`
    // (each op fires once per spatial tile of the eventual output).
    let compute_latency = |output_id| {
        let mut latency = PerformanceMetric::zero();
        let mut stack = vec![output_id];
        while let Some(tensor_id) = stack.pop() {
            if input_tensor_ids.contains(&tensor_id)
                || tensor_id != output_id && output_tensor_ids.contains(&tensor_id)
            {
                continue;
            }

            let producer_id = graph.producer_id_of(tensor_id).unwrap();
            let producer_op = &graph.producer_of(tensor_id).unwrap();
            latency += PerformanceMetric::from_compute_cost(producer_op.base_cost.into());
            stack.extend(graph.input_ids_for(producer_id));
        }
        latency
    };

    let output_memory_cost = |output_id: TensorId, tile_index: (i64, i64)| -> Fraction {
        if retained_tensor_ids.contains(&output_id) {
            Fraction::from(0i64)
        } else {
            let position = (tile_index.0 * tile_h, tile_index.1 * tile_w);
            let tensor = &graph.tensors()[output_id.0];
            let evict_bytes = i64::min(tile_h, tensor.height - position.0)
                * i64::min(tile_w, tensor.width - position.1);
            Fraction::from(evict_bytes) / bandwidth
        }
    };

    let mut cached_inputs = ResidencySet::new();
    let mut latencies: HashMap<TensorId, Vec<PerformanceMetric>> = HashMap::new();
    for &output_id in output_tensor_ids.iter() {
        let output_producer = &graph.producer_of(output_id).unwrap();
        let output_producer_id = graph.producer_id_of(output_id).unwrap();
        let reduction_size = match output_producer.kind {
            OperationType::MatMul => {
                graph.tensors()[graph.input_ids_for(output_producer_id)[0].0].width
            }
            OperationType::Pointwise => 1,
        };
        let reduction_counts = ceil_div(reduction_size, tile_k);
        let mut output_latencies: Vec<PerformanceMetric> = vec![];
        let compute_total = compute_latency(output_id).compute_cost;

        for m_tile_idx in 0..tile_counts.0 {
            for n_step in 0..tile_counts.1 {
                let n_tile_idx = if m_tile_idx % 2 == 0 {
                    n_step
                } else {
                    tile_counts.1 - 1 - n_step
                };
                for k_tile_idx in 0..reduction_counts {
                    let names = slice_names_for_output(
                        subgraph,
                        output_id,
                        tile_size,
                        (m_tile_idx, n_tile_idx, k_tile_idx),
                    );
                    let reduction_tile_size =
                        i64::min(tile_k, reduction_size - k_tile_idx * tile_k);
                    let latency = PerformanceMetric::from_memory_cost(input_memory_cost(
                        &names,
                        &cached_inputs,
                    )) + PerformanceMetric::from_compute_cost(
                        compute_total / reduction_size * reduction_tile_size,
                    );
                    cached_inputs.replace(names.iter().copied());
                    output_latencies.push(latency);
                }
                *output_latencies.last_mut().unwrap() += PerformanceMetric::from_memory_cost(
                    output_memory_cost(output_id, (m_tile_idx, n_tile_idx)),
                );
            }
        }
        latencies.insert(output_id, output_latencies);
    }

    latencies
}

pub fn total_latency(latencies: &HashMap<TensorId, Vec<PerformanceMetric>>) -> Fraction {
    latencies
        .values()
        .flat_map(|metrics| metrics.iter().map(|metric| metric.latency()))
        .sum()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use fraction::Fraction;

    use super::{subgraph_latency, total_latency};
    use crate::graph::{ComputationGraph, TensorId};
    use crate::testutil::{load_input, subgraph};

    // Official Example 1, Strategy A: two separate pointwise subgraphs, each
    // producing its own spill to slow memory. Expected per-op: 32768.
    #[test]
    fn example1_strategy_a_sequential() {
        let input = load_input("official_example1.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg0 = subgraph(&graph, [0]);
        let sg1 = subgraph(&graph, [1]);

        let lat0 = subgraph_latency(&device, &sg0, (128, 128, 1), &[]);
        let lat1 = subgraph_latency(&device, &sg1, (128, 128, 1), &[]);

        assert_eq!(lat0.len(), 1);
        assert_eq!(lat1.len(), 1);
        assert_eq!(total_latency(&lat0), Fraction::new(32768u64, 10u64));
        assert_eq!(total_latency(&lat1), Fraction::new(32768u64, 10u64));
        assert_eq!(
            total_latency(&lat0) + total_latency(&lat1),
            Fraction::new(65536u64, 10u64)
        );
    }

    // Official Example 1, Strategy B: single merged subgraph eliminates the
    // intermediate spill of Strategy A. Expected total: 3276.8.
    #[test]
    fn example1_strategy_b_merged() {
        let input = load_input("official_example1.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0, 1]);
        let latencies = subgraph_latency(&device, &sg, (128, 128, 1), &[]);

        assert_eq!(latencies.len(), 1);
        assert_eq!(total_latency(&latencies), Fraction::new(32768u64, 10u64));
    }

    // Official Example 1, Strategy C: merged subgraph with small (64,64) tiles
    // pushes the op into compute-bound territory. Expected total: 4400.
    #[test]
    fn example1_strategy_c_small_tiles() {
        let input = load_input("official_example1.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0, 1]);
        let latencies = subgraph_latency(&device, &sg, (64, 64, 1), &[]);

        assert_eq!(total_latency(&latencies), Fraction::new(44000u64, 10u64));
    }

    // Official Example 2, Strategy A: 256x256 tensors tiled at 128x128. Four
    // spatial tiles per op. Expected per-op: 13107.2.
    #[test]
    fn example2_strategy_a_sequential() {
        let input = load_input("official_example2.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg0 = subgraph(&graph, [0]);
        let sg1 = subgraph(&graph, [1]);

        let lat0 = subgraph_latency(&device, &sg0, (128, 128, 1), &[]);
        let lat1 = subgraph_latency(&device, &sg1, (128, 128, 1), &[]);

        assert_eq!(total_latency(&lat0), Fraction::new(131072u64, 10u64));
        assert_eq!(total_latency(&lat1), Fraction::new(131072u64, 10u64));
    }

    // Official Example 2, Strategy B: merging eliminates the intermediate
    // 256x256 spill. Expected total: 13107.2.
    #[test]
    fn example2_strategy_b_merged() {
        let input = load_input("official_example2.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0, 1]);
        let latencies = subgraph_latency(&device, &sg, (128, 128, 1), &[]);

        assert_eq!(total_latency(&latencies), Fraction::new(131072u64, 10u64));
    }

    // Retaining the subgraph's output tensor removes its spill traffic.
    // For Example 1B (merged) with t2 retained, expected total: 1638.4.
    #[test]
    fn retained_output_skips_output_traffic() {
        let input = load_input("official_example1.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0, 1]);
        let latencies = subgraph_latency(&device, &sg, (128, 128, 1), &[TensorId(2)]);

        assert_eq!(total_latency(&latencies), Fraction::new(16384u64, 10u64));
    }

    // Official Example 3, Strategy A: diamond graph with three separate
    // pointwise subgraphs. Expected per-op: 3276.8, 3276.8, 4915.2
    // (total 11468.8).
    #[test]
    fn example3_strategy_a_spilling() {
        let input = load_input("official_example3.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg0 = subgraph(&graph, [0]);
        let sg1 = subgraph(&graph, [1]);
        let sg2 = subgraph(&graph, [2]);

        let lat0 = subgraph_latency(&device, &sg0, (128, 128, 1), &[]);
        let lat1 = subgraph_latency(&device, &sg1, (128, 128, 1), &[]);
        let lat2 = subgraph_latency(&device, &sg2, (128, 128, 1), &[]);

        assert_eq!(total_latency(&lat0), Fraction::new(32768u64, 10u64));
        assert_eq!(total_latency(&lat1), Fraction::new(32768u64, 10u64));
        assert_eq!(total_latency(&lat2), Fraction::new(49152u64, 10u64));
        let total = total_latency(&lat0) + total_latency(&lat1) + total_latency(&lat2);
        assert_eq!(total, Fraction::new(114688u64, 10u64));
    }

    // Official Example 3, Strategy C: selective residency. Subgraphs [op0]
    // (retain t1) and [op1, op2]. Expected: 1638.4 + 3000 = 4638.4.
    #[test]
    fn example3_strategy_c_selective_residency() {
        let input = load_input("official_example3.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg_0 = subgraph(&graph, [0]);
        let sg_12 = subgraph(&graph, [1, 2]);

        let lat_0 = subgraph_latency(&device, &sg_0, (128, 128, 1), &[TensorId(1)]);
        let lat_12 = subgraph_latency(&device, &sg_12, (128, 128, 1), &[TensorId(1)]);

        assert_eq!(total_latency(&lat_0), Fraction::new(16384u64, 10u64));
        assert_eq!(total_latency(&lat_12), Fraction::new(30000u64, 10u64));
        assert_eq!(
            total_latency(&lat_0) + total_latency(&lat_12),
            Fraction::new(46384u64, 10u64)
        );
    }

    // Official Example 4, Strategy A: single MatMul tiled at (64,64,128) in
    // snake order. Snake reuses one extra LHS row strip at each m-row turn
    // versus raster's 7096.
    #[test]
    fn example4_strategy_a_snake() {
        let input = load_input("official_example4.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0]);
        let latencies = subgraph_latency(&device, &sg, (64, 64, 128), &[]);

        assert_eq!(latencies.len(), 1);
        assert_eq!(total_latency(&latencies), "6548".parse().unwrap());
    }

    // Official Example 5, Strategy B: chained MatMuls merged into one
    // subgraph with split-K tiling (128,128,32). Expected total: 6915.2.
    #[test]
    fn example5_strategy_b_split_k() {
        let input = load_input("official_example5.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0, 1]);
        let latencies = subgraph_latency(&device, &sg, (128, 128, 32), &[]);

        assert_eq!(latencies.len(), 1);
        assert_eq!(total_latency(&latencies), Fraction::new(69152u64, 10u64));
    }

    #[test]
    fn example5_unfused() {
        let input = load_input("official_example5.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg0 = subgraph(&graph, [0]);
        let latencies0 = subgraph_latency(&device, &sg0, (128, 128, 32), &[]);
        let sg1 = subgraph(&graph, [1]);
        let latencies1 = subgraph_latency(&device, &sg1, (128, 128, 32), &[]);

        assert_eq!(latencies0.len(), 1);
        assert_eq!(
            total_latency(&latencies0),
            Fraction::from_str("4915.2").unwrap()
        );
        assert_eq!(latencies1.len(), 1);
        assert_eq!(
            total_latency(&latencies1),
            Fraction::from_str("4915.2").unwrap()
        );
    }
}
