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
    let tile_counts = {
        let output_id = output_tensor_ids.iter().next().unwrap();
        let output = &graph.tensors()[output_id.0];
        (
            ceil_div(output.height, tile_size.0),
            ceil_div(output.width, tile_size.1),
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
            let output_tile_position = (tile_index.0 * tile_size.0, tile_index.1 * tile_size.1);
            let tensor = &graph.tensors()[output_id.0];
            let output_tile_size = i64::min(tile_size.0, tensor.height - output_tile_position.0)
                * i64::min(tile_size.1, tensor.width - output_tile_position.1);
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
        let reduction_counts = ceil_div(reduction_size, tile_size.2);
        let mut output_latencies = vec![];
        let compute_cost_per_spatial_tile = compute_latency(output_id).compute_cost;
        for m_tile_idx in 0..tile_counts.0 {
            for n_tile_idx in 0..tile_counts.1 {
                for k_tile_idx in 0..reduction_counts {
                    let input_tiles = input_tiles_for_output(
                        subgraph,
                        output_id,
                        tile_size,
                        (m_tile_idx, n_tile_idx, k_tile_idx),
                    );
                    let reduction_tile_size =
                        i64::min(tile_size.2, reduction_size - k_tile_idx * tile_size.2);
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

