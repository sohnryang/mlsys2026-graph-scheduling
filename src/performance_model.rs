use std::{
    cmp::{self, Ordering},
    collections::{HashMap, HashSet},
    ops::{Add, AddAssign},
};

use fraction::Fraction;

use crate::{
    graph::{OperationType, Subgraph, TensorId},
    input_format::DeviceParameters,
    tiling::ceil_div,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TensorSlice {
    position: (i64, i64),
    shape: (i64, i64),
}

impl TensorSlice {
    fn size(&self) -> i64 {
        self.shape.0 * self.shape.1
    }
}

fn input_tiles_for_output(
    subgraph: &Subgraph<'_>,
    output_id: TensorId,
    tile_size: (i64, i64, i64),
    tile_index: (i64, i64, i64),
) -> HashMap<TensorId, Vec<TensorSlice>> {
    let input_tensor_ids = subgraph.input_tensor_ids();
    let graph = subgraph.parent();
    let output_tensor = &graph.tensors()[output_id.0];
    let output_tile_position = (tile_index.0 * tile_size.0, tile_index.1 * tile_size.1);
    let output_tile_shape = (
        i64::min(tile_size.0, output_tensor.height - output_tile_position.0),
        i64::min(tile_size.1, output_tensor.width - output_tile_position.1),
    );
    let mut stack = vec![(
        output_id,
        TensorSlice {
            position: output_tile_position,
            shape: output_tile_shape,
        },
    )];

    let mut input_tiles = HashMap::new();
    while let Some((tensor_id, tensor_slice)) = stack.pop() {
        if input_tensor_ids.contains(&tensor_id) {
            input_tiles
                .entry(tensor_id)
                .or_insert(vec![])
                .push(tensor_slice);
            continue;
        }

        let producer_id = graph.producer_id_of(tensor_id).unwrap();
        let producer_op = &graph.producer_of(tensor_id).unwrap();
        let input_ids = graph.input_ids_for(producer_id);
        match producer_op.kind {
            OperationType::Pointwise => {
                for &input_id in input_ids {
                    stack.push((input_id, tensor_slice));
                }
            }
            OperationType::MatMul => {
                let input0_id = input_ids[0];
                let input0 = &graph.tensors()[input0_id.0];
                let input1_id = input_ids[1];

                let reduction_index = if tensor_id == output_id {
                    tile_index.2 * tile_size.2
                } else {
                    0
                };
                let reduction_size = if tensor_id == output_id {
                    i64::min(tile_size.2, input0.width - reduction_index)
                } else {
                    input0.width
                };
                stack.push((
                    input0_id,
                    TensorSlice {
                        position: (tensor_slice.position.0, reduction_index),
                        shape: (tensor_slice.shape.0, reduction_size),
                    },
                ));
                stack.push((
                    input1_id,
                    TensorSlice {
                        position: (reduction_index, tensor_slice.position.1),
                        shape: (reduction_size, tensor_slice.shape.1),
                    },
                ));
            }
        };
    }
    input_tiles
}

pub fn subgraph_latency(
    device_params: &DeviceParameters,
    subgraph: &Subgraph<'_>,
    tile_size: (i64, i64, i64),
    retained_tensor_ids: &[TensorId],
) -> HashMap<TensorId, Vec<PerformanceMetric>> {
    let retained_tensor_ids = retained_tensor_ids.iter().copied().collect::<HashSet<_>>();
    let input_tensor_ids = subgraph.input_tensor_ids();
    let output_tensor_ids = {
        let mut v = subgraph.output_tensor_ids().into_iter().collect::<Vec<_>>();
        v.sort();
        v
    };

    let graph = subgraph.parent();
    let (tile_w, tile_h, tile_k) = tile_size;
    let tile_counts = {
        let output_id = output_tensor_ids.iter().next().unwrap();
        let output = &graph.tensors()[output_id.0];
        (
            ceil_div(output.height, tile_h),
            ceil_div(output.width, tile_w),
        )
    };

    let input_memory_cost =
        |input_tiles: &HashMap<TensorId, Vec<TensorSlice>>,
         cached_inputs: &HashSet<(TensorId, TensorSlice)>| {
            let input_traffic: i64 = input_tiles
                .iter()
                .map(|(&tensor_id, tensor_slices)| -> i64 {
                    tensor_slices
                        .iter()
                        .filter_map(|&tensor_slice| {
                            if retained_tensor_ids.contains(&tensor_id)
                                || cached_inputs.contains(&(tensor_id, tensor_slice))
                            {
                                None
                            } else {
                                Some(tensor_slice.size())
                            }
                        })
                        .sum()
                })
                .sum();
            Fraction::from(input_traffic) / device_params.slow_memory_bandwidth
        };
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
    let output_memory_cost = |output_id, tile_index: (i64, i64)| {
        if retained_tensor_ids.contains(&output_id) {
            Fraction::from(0i64)
        } else {
            let output_tile_position = (tile_index.0 * tile_h, tile_index.1 * tile_w);
            let tensor = &graph.tensors()[output_id.0];
            let output_tile_size = i64::min(tile_h, tensor.height - output_tile_position.0)
                * i64::min(tile_w, tensor.width - output_tile_position.1);
            Fraction::from(output_tile_size) / device_params.slow_memory_bandwidth
        }
    };

    let mut cached_inputs = HashSet::new();
    let mut latencies = HashMap::new();
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
        let mut output_latencies = vec![];
        let compute_cost_per_spatial_tile = compute_latency(output_id).compute_cost;
        for m_tile_idx in 0..tile_counts.0 {
            for n_tile_idx in 0..tile_counts.1 {
                for k_tile_idx in 0..reduction_counts {
                    let input_tiles = input_tiles_for_output(
                        subgraph,
                        output_id,
                        (tile_h, tile_w, tile_k),
                        (m_tile_idx, n_tile_idx, k_tile_idx),
                    );
                    let reduction_tile_size =
                        i64::min(tile_k, reduction_size - k_tile_idx * tile_k);
                    let latency = PerformanceMetric::from_memory_cost(input_memory_cost(
                        &input_tiles,
                        &cached_inputs,
                    )) + PerformanceMetric::from_compute_cost(
                        compute_cost_per_spatial_tile / reduction_size * reduction_tile_size,
                    );
                    cached_inputs = input_tiles
                        .into_iter()
                        .flat_map(|(tensor_id, tensor_slices)| {
                            tensor_slices
                                .into_iter()
                                .map(move |tensor_slice| (tensor_id, tensor_slice))
                        })
                        .collect();
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
    // raster order. Expected total: 7096.
    #[test]
    fn example4_strategy_a_raster() {
        let input = load_input("official_example4.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0]);
        let latencies = subgraph_latency(&device, &sg, (64, 64, 128), &[]);

        assert_eq!(latencies.len(), 1);
        assert_eq!(total_latency(&latencies), Fraction::new(70960u64, 10u64));
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
