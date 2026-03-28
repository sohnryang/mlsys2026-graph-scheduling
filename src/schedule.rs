use std::{collections::HashSet, iter};

use crate::graph::{ComputationGraph, Subgraph};

pub fn extract_convex_subgraphs(graph: &ComputationGraph) -> HashSet<Subgraph<'_>> {
    let mut execution_states = HashSet::from([Subgraph::from_nodes(graph, iter::empty())]);
    let topological_order = graph.topological_sort();
    let mut stack = vec![Subgraph::from_nodes(graph, iter::empty())];
    while let Some(execution_state) = stack.pop() {
        for &operation_id in topological_order.iter() {
            if execution_state.contains(operation_id) {
                continue;
            }

            let dependency_satisfied = graph
                .input_ids_for(operation_id)
                .iter()
                .filter_map(|&tensor_id| graph.producer_id_of(tensor_id))
                .all(|producer_id| execution_state.contains(producer_id));
            if !dependency_satisfied {
                continue;
            }

            let mut next_executed_subgraph = execution_state.clone();
            next_executed_subgraph.insert(operation_id);
            if execution_states.contains(&next_executed_subgraph) {
                continue;
            }

            execution_states.insert(next_executed_subgraph.clone());
            stack.push(next_executed_subgraph);
        }
    }

    let mut convex_subgraphs = HashSet::new();
    for state0 in execution_states.iter() {
        for state1 in execution_states.iter() {
            if !state0.is_subset(state1) {
                continue;
            }
            let convex_subgraph = state1.subtract(state0);
            if convex_subgraph.components() == 1 {
                convex_subgraphs.insert(convex_subgraph);
            }
        }
    }
    convex_subgraphs
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::extract_convex_subgraphs;
    use crate::{
        graph::{ComputationGraph, OperationId, OperationType, Subgraph, TensorId},
        input_format::{DeviceParameters, InputFormat},
    };

    fn make_input(
        inputs: Vec<Vec<TensorId>>,
        outputs: Vec<Vec<TensorId>>,
        num_tensors: usize,
    ) -> InputFormat {
        let num_ops = inputs.len();
        InputFormat {
            widths: vec![1; num_tensors],
            heights: vec![1; num_tensors],
            inputs,
            outputs,
            base_costs: vec![1; num_ops],
            op_types: vec![OperationType::Pointwise; num_ops],
            device_parameters: DeviceParameters {
                fast_memory_capacity: 1,
                slow_memory_bandwidth: 1,
                native_granularity: (1, 1),
            },
        }
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // convex subgraphs: {
    //   {op0},
    //   {op1},
    //   {op2},
    //   {op3},
    //   {op0, op1},
    //   {op0, op2},
    //   {op1, op3},
    //   {op2, op3},
    //   {op0, op1, op2},
    //   {op1, op2, op3},
    //   {op0, op1, op2, op3},
    // }
    #[test]
    fn skip_connection_convex_subgraphs() {
        let input = make_input(
            vec![
                vec![TensorId(0)],
                vec![TensorId(1)],
                vec![TensorId(2)],
                vec![TensorId(3), TensorId(4)],
            ],
            vec![
                vec![TensorId(1), TensorId(2)],
                vec![TensorId(3)],
                vec![TensorId(4)],
                vec![TensorId(5)],
            ],
            6,
        );
        let graph = ComputationGraph::new(&input);
        let convex_subgraphs = extract_convex_subgraphs(&graph);
        let expected = HashSet::from([
            Subgraph::from_nodes(&graph, [OperationId(0)]),
            Subgraph::from_nodes(&graph, [OperationId(1)]),
            Subgraph::from_nodes(&graph, [OperationId(2)]),
            Subgraph::from_nodes(&graph, [OperationId(3)]),
            Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]),
            Subgraph::from_nodes(&graph, [OperationId(0), OperationId(2)]),
            Subgraph::from_nodes(&graph, [OperationId(1), OperationId(3)]),
            Subgraph::from_nodes(&graph, [OperationId(2), OperationId(3)]),
            Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1), OperationId(2)]),
            Subgraph::from_nodes(&graph, [OperationId(1), OperationId(2), OperationId(3)]),
            Subgraph::from_nodes(
                &graph,
                [
                    OperationId(0),
                    OperationId(1),
                    OperationId(2),
                    OperationId(3),
                ],
            ),
        ]);
        assert_eq!(convex_subgraphs, expected);
    }
}
