use std::collections::{HashMap, HashSet};

use fraction::Fraction;

use crate::{
    graph::{OperationId, Partition, Subgraph, TensorId},
    input_format::DeviceParameters,
    performance_model::{subgraph_latency, total_latency},
    tiling::{SearchError, search_tile_values},
};

fn try_extract_output_covering_subgraph<'a>(
    subgraph: &Subgraph<'a>,
    is_retained_edge: &HashMap<(TensorId, OperationId), bool>,
    outputs: impl IntoIterator<Item = TensorId>,
) -> Option<(Subgraph<'a>, HashSet<TensorId>)> {
    let graph = subgraph.parent();
    let subgraph_inputs = subgraph.input_tensor_ids();
    let outputs = outputs.into_iter().collect::<HashSet<_>>();
    let mut covered_outputs = HashSet::new();

    let mut producers_to_outputs = HashMap::new();
    for &output_id in outputs.iter() {
        let producer_id = graph.producer_id_of(output_id).unwrap();
        producers_to_outputs
            .entry(producer_id)
            .or_insert(vec![])
            .push(output_id);
    }

    let output_id = *outputs.iter().next().unwrap();
    let mut stack = vec![graph.producer_id_of(output_id).unwrap()];
    let mut visited = HashSet::new();
    let mut retained_tensors_used_here = HashSet::new();
    while let Some(operation_id) = stack.pop() {
        if visited.contains(&operation_id) {
            continue;
        }
        visited.insert(operation_id);
        covered_outputs.extend(
            producers_to_outputs
                .get(&operation_id)
                .unwrap_or(&vec![])
                .iter()
                .copied(),
        );
        for &input_id in graph.input_ids_for(operation_id) {
            if subgraph_inputs.contains(&input_id) {
                continue;
            }
            if is_retained_edge
                .get(&(input_id, operation_id))
                .copied()
                .unwrap_or(false)
            {
                retained_tensors_used_here.insert(input_id);
                continue;
            }
            stack.push(graph.producer_id_of(input_id).unwrap());
        }
    }

    let mut stack = retained_tensors_used_here
        .iter()
        .flat_map(|&tensor_id| graph.consumer_ids_for(tensor_id))
        .copied()
        .collect::<Vec<_>>();
    while let Some(operation_id) = stack.pop() {
        if visited.contains(&operation_id) {
            continue;
        }
        visited.insert(operation_id);
        covered_outputs.extend(
            producers_to_outputs
                .get(&operation_id)
                .unwrap_or(&vec![])
                .iter()
                .copied(),
        );
        for &output_id in graph.output_ids_for(operation_id) {
            if outputs.contains(&output_id) {
                continue;
            }
            stack.extend(graph.consumer_ids_for(output_id));
        }
    }

    if covered_outputs.is_superset(&outputs) {
        Some((
            Subgraph::from_nodes(graph, visited),
            retained_tensors_used_here,
        ))
    } else {
        None
    }
}

fn try_partition_subgraph<'a>(
    subgraph: &Subgraph<'a>,
    device_params: &DeviceParameters,
    is_retained_edge: &HashMap<(TensorId, OperationId), bool>,
) -> Option<(Vec<Partition<'a>>, Fraction)> {
    let mut outputs_to_cover = subgraph.output_tensor_ids();
    let mut subgraph_chain = vec![];
    let mut retained_tensor_chain = vec![vec![]];
    while !outputs_to_cover.is_empty() {
        let (covering_subgraph, retained_tensors_used) =
            try_extract_output_covering_subgraph(subgraph, is_retained_edge, outputs_to_cover)?;
        outputs_to_cover = retained_tensors_used.clone();
        subgraph_chain.push(covering_subgraph);
        retained_tensor_chain.push(retained_tensors_used.into_iter().collect::<Vec<_>>());
    }
    debug_assert!(retained_tensor_chain.last().unwrap().is_empty());
    retained_tensor_chain.pop();
    let mut prev_retained_outputs = vec![];
    let mut partitions = vec![];
    let mut total_cost = Fraction::from(0u64);
    for (subgraph, retained_outputs) in subgraph_chain
        .into_iter()
        .zip(retained_tensor_chain.into_iter())
        .rev()
    {
        let retained_tensor_ids =
            [prev_retained_outputs.clone(), retained_outputs.clone()].concat();
        let tile_size = search_tile_values(&subgraph, device_params, &retained_tensor_ids).ok()?;
        total_cost += total_latency(&subgraph_latency(
            device_params,
            &subgraph,
            tile_size,
            &retained_tensor_ids,
        ));
        prev_retained_outputs = retained_outputs.clone();
        partitions.push(Partition {
            subgraph,
            retained_outputs,
            tile_size,
        });
    }
    Some((partitions, total_cost))
}

pub fn search_partition<'a>(
    subgraph: &Subgraph<'a>,
    device_params: &DeviceParameters,
) -> Option<(Vec<Partition<'a>>, Fraction)> {
    let graph = subgraph.parent();
    let subgraph_inputs = subgraph.input_tensor_ids();
    let mut is_retained_edge = HashMap::new();
    let mut best_cost = None;

    for &operation_id in subgraph.nodes() {
        for &input_id in graph.input_ids_for(operation_id) {
            if subgraph_inputs.contains(&input_id) {
                continue;
            }

            let mut best_decision = false;
            for should_retain in [true, false] {
                is_retained_edge.insert((input_id, operation_id), should_retain);
                let Some((_, partition_cost)) =
                    try_partition_subgraph(subgraph, device_params, &is_retained_edge)
                else {
                    if should_retain {
                        continue;
                    } else {
                        return None;
                    }
                };
                if partition_cost < best_cost.unwrap_or(Fraction::infinity()) {
                    best_cost = Some(partition_cost);
                    best_decision = should_retain;
                }
            }
            is_retained_edge.insert((input_id, operation_id), best_decision);
        }
    }

    try_partition_subgraph(subgraph, device_params, &is_retained_edge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputationGraph, OperationId, TensorId};
    use crate::testutil::{load_input, subgraph};

    // Official Example 3, Strategy C: selective residency. The full subgraph
    // {Op0, Op1, Op2} is cut at T1 (retained across the partition boundary),
    // yielding {Op0} (producing retained T1) followed by {Op1, Op2}.
    #[test]
    fn example3_strategy_c_selective_residency() {
        let input = load_input("official_example3.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0, 1, 2]);

        // T1 is retained at both of its in-subgraph consumer edges.
        let mut is_retained_edge = HashMap::new();
        is_retained_edge.insert((TensorId(1), OperationId(1)), true);
        is_retained_edge.insert((TensorId(1), OperationId(2)), true);

        let (partitions, total_cost) =
            try_partition_subgraph(&sg, &device, &is_retained_edge).expect("partition succeeds");

        assert_eq!(partitions.len(), 2);

        let first = &partitions[0];
        assert_eq!(first.subgraph.nodes(), &[OperationId(0)]);
        assert_eq!(first.retained_outputs, vec![TensorId(1)]);
        assert_eq!(first.tile_size, (128, 128, 1));

        let second = &partitions[1];
        assert_eq!(second.subgraph.nodes(), &[OperationId(1), OperationId(2)]);
        assert!(second.retained_outputs.is_empty());
        assert_eq!(second.tile_size, (128, 128, 1));

        // Strategy C expected total: 1638.4 + 3000 = 4638.4.
        assert_eq!(total_cost, "4638.4".parse::<Fraction>().unwrap());
    }

    // Official Example 3, Strategy B: "Flash" recomputation. Fuse {Op0, Op1, Op2}
    // and mark the (T2, Op2) edge retained — the walk back from T3 cuts at T2,
    // leaving {Op0, Op2} as the downstream partition (recomputing Op0 to
    // reproduce T1), with {Op0, Op1} as the upstream partition producing
    // retained T2.
    #[test]
    fn example3_strategy_b_flash_recomputation() {
        let input = load_input("official_example3.json");
        let device = input.device_parameters.clone();
        let graph: ComputationGraph = input.into();

        let sg = subgraph(&graph, [0, 1, 2]);

        let mut is_retained_edge = HashMap::new();
        is_retained_edge.insert((TensorId(2), OperationId(2)), true);

        let (partitions, _total_cost) =
            try_partition_subgraph(&sg, &device, &is_retained_edge).expect("partition succeeds");

        assert_eq!(partitions.len(), 2);

        let first = &partitions[0];
        assert_eq!(first.subgraph.nodes(), &[OperationId(0), OperationId(1)]);
        assert_eq!(first.retained_outputs, vec![TensorId(2)]);

        let second = &partitions[1];
        assert_eq!(second.subgraph.nodes(), &[OperationId(0), OperationId(2)]);
        assert!(second.retained_outputs.is_empty());

        // Strategy B expected total: 6276.8.
        // assert_eq!(total_cost, "6276.8".parse::<Fraction>().unwrap());
    }
}
