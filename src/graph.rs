use std::collections::HashMap;

use serde::Deserialize;

use crate::input_format::InputFormat;

#[derive(Clone, Copy, Debug, Deserialize)]
pub enum OperationType {
    MatMul,
    Pointwise,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub struct TensorId(pub usize);

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: TensorId,
    pub width: i64,
    pub height: i64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub struct OperationId(pub usize);

#[derive(Clone, Debug)]
pub struct Operation {
    pub kind: OperationType,
    pub base_cost: i64,
}

#[derive(Clone, Debug)]
pub struct ComputationGraph {
    operations: Vec<Operation>,
    tensors: Vec<Tensor>,
    outputs: HashMap<OperationId, Vec<TensorId>>,
    users: HashMap<TensorId, Vec<OperationId>>,
}

impl ComputationGraph {
    pub fn new(input: &InputFormat) -> Self {
        assert_eq!(
            input.widths.len(),
            input.heights.len(),
            "widths and heights must have equal length"
        );

        let num_ops = input.inputs.len();
        assert_eq!(
            input.outputs.len(),
            num_ops,
            "inputs and outputs must have equal number of operations"
        );
        assert_eq!(
            input.base_costs.len(),
            num_ops,
            "base_costs length must match operation count"
        );
        assert_eq!(
            input.op_types.len(),
            num_ops,
            "op_types length must match operation count"
        );

        let tensors = input
            .widths
            .iter()
            .copied()
            .zip(input.heights.iter().copied())
            .enumerate()
            .map(|(tensor_idx, (width, height))| Tensor {
                id: TensorId(tensor_idx),
                width,
                height,
            })
            .collect();

        let operations = input
            .op_types
            .iter()
            .copied()
            .zip(input.base_costs.iter().copied())
            .map(|(kind, base_cost)| Operation { kind, base_cost })
            .collect();

        let num_tensors = input.widths.len();
        for op_tensors in input.inputs.iter().chain(input.outputs.iter()) {
            for &tensor_id in op_tensors {
                assert!(
                    tensor_id.0 < num_tensors,
                    "tensor id {} is out of range 0..{}",
                    tensor_id.0,
                    num_tensors
                );
            }
        }

        let outputs = input
            .outputs
            .iter()
            .cloned()
            .enumerate()
            .map(|(op_idx, op_outputs)| (OperationId(op_idx), op_outputs))
            .collect();

        let mut tensor_users = HashMap::new();
        for (op_id, input_ids) in input.inputs.iter().enumerate() {
            for &input_id in input_ids {
                tensor_users
                    .entry(input_id)
                    .or_insert(vec![])
                    .push(OperationId(op_id));
            }
        }

        Self {
            operations,
            tensors,
            outputs,
            users: tensor_users,
        }
    }

    pub fn outputs(&self) -> &HashMap<OperationId, Vec<TensorId>> {
        &self.outputs
    }

    pub fn output_ids_for(&self, operation_id: OperationId) -> &[TensorId] {
        self.outputs
            .get(&operation_id)
            .expect(format!("{operation_id:?} does not exist").as_str())
    }

    pub fn outputs_for(&self, operation_id: OperationId) -> Vec<Tensor> {
        self.outputs[&operation_id]
            .iter()
            .map(|id| self.tensors[id.0].clone())
            .collect()
    }

    pub fn users(&self) -> &HashMap<TensorId, Vec<OperationId>> {
        &self.users
    }

    pub fn user_ids_for(&self, tensor_id: TensorId) -> &[OperationId] {
        self.users
            .get(&tensor_id)
            .expect(format!("{tensor_id:?} does not exist").as_str())
    }

    pub fn users_for(&self, tensor_id: TensorId) -> Vec<Operation> {
        self.users[&tensor_id]
            .iter()
            .map(|id| self.operations[id.0].clone())
            .collect()
    }

    pub fn topological_sort(&self) -> Vec<OperationId> {
        let num_ops = self.operations.len();

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum State {
            Unvisited,
            Visiting,
            Visited,
        }
        let mut state = vec![State::Unvisited; num_ops];
        let mut post_order = Vec::with_capacity(num_ops);

        for start_idx in 0..num_ops {
            if state[start_idx] != State::Unvisited {
                continue;
            }

            let mut stack = vec![OperationId(start_idx)];
            while let Some(&operation_id) = stack.last() {
                match state[operation_id.0] {
                    State::Unvisited => {
                        state[operation_id.0] = State::Visiting;
                        self.outputs[&operation_id]
                            .iter()
                            .flat_map(|tensor_id| {
                                self.users
                                    .get(tensor_id)
                                    .map(|v| v.iter())
                                    .unwrap_or_default()
                            })
                            .for_each(|&user_id| {
                                if state[user_id.0] == State::Unvisited {
                                    stack.push(user_id);
                                }
                            });
                    }
                    State::Visiting => {
                        state[operation_id.0] = State::Visited;
                        post_order.push(operation_id);
                        stack.pop();
                    }
                    State::Visited => {
                        stack.pop();
                    }
                };
            }
        }

        post_order.reverse();
        post_order
    }
}

#[cfg(test)]
mod tests {
    use super::{ComputationGraph, OperationId, OperationType, TensorId};
    use crate::input_format::InputFormat;

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
            fast_memory_capacity: 1,
            slow_memory_bandwidth: 1,
            native_granularity: (1, 1),
        }
    }

    fn assert_is_topological_order(graph: &ComputationGraph, order: &[OperationId]) {
        let num_ops = graph.operations.len();
        let mut position = vec![None; num_ops];

        for (idx, op_id) in order.iter().copied().enumerate() {
            assert!(op_id.0 < num_ops, "operation id out of range: {}", op_id.0);
            assert!(
                position[op_id.0].is_none(),
                "operation appears more than once: {:?}",
                op_id
            );
            position[op_id.0] = Some(idx);
        }
        assert!(
            position.iter().all(|p| p.is_some()),
            "some operations are missing in the topological order"
        );

        for (op_idx, op_outputs) in graph.outputs() {
            for tensor_id in op_outputs {
                if let Some(users) = graph.users().get(tensor_id) {
                    for &user in users {
                        assert!(
                            position[op_idx.0] < position[user.0],
                            "dependency violated: {:?} must come before {:?}",
                            op_idx,
                            user
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn topological_sort_empty_graph() {
        let input = make_input(vec![], vec![], 0);
        let graph = ComputationGraph::new(&input);

        assert_eq!(graph.topological_sort(), vec![]);
    }

    #[test]
    fn topological_sort_linear_chain() {
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t2 = TensorId(2);

        let input = make_input(
            vec![vec![], vec![t0], vec![t1]],
            vec![vec![t0], vec![t1], vec![t2]],
            3,
        );
        let graph = ComputationGraph::new(&input);

        assert_eq!(
            graph.topological_sort(),
            vec![OperationId(0), OperationId(1), OperationId(2)]
        );
    }

    #[test]
    fn topological_sort_branching_dag() {
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t2 = TensorId(2);
        let t3 = TensorId(3);
        let t4 = TensorId(4);

        let input = make_input(
            vec![vec![], vec![t0], vec![t1], vec![t2, t3]],
            vec![vec![t0, t1], vec![t2], vec![t3], vec![t4]],
            5,
        );
        let graph = ComputationGraph::new(&input);
        let order = graph.topological_sort();

        assert_eq!(order.len(), 4);
        assert_is_topological_order(&graph, &order);
    }

    #[test]
    fn topological_sort_disconnected_components() {
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t2 = TensorId(2);
        let t3 = TensorId(3);

        let input = make_input(
            vec![vec![], vec![t0], vec![], vec![t2]],
            vec![vec![t0], vec![t1], vec![t2], vec![t3]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let order = graph.topological_sort();

        assert_eq!(order.len(), 4);
        assert_is_topological_order(&graph, &order);
    }
}
