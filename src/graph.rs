use std::cell::OnceCell;
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
pub struct TopologicalOrder {
    order: Vec<OperationId>,
    /// `position[op.0]` is the index of `op` in `order`.
    position: Vec<usize>,
}

impl TopologicalOrder {
    /// Collects `iter` into a `Vec<OperationId>` sorted by topological position.
    pub fn sort(&self, iter: impl IntoIterator<Item = OperationId>) -> Vec<OperationId> {
        let mut ops: Vec<OperationId> = iter.into_iter().collect();
        ops.sort_by_key(|op| self.position[op.0]);
        ops
    }
}

impl std::ops::Deref for TopologicalOrder {
    type Target = [OperationId];
    fn deref(&self) -> &[OperationId] {
        &self.order
    }
}

impl IntoIterator for TopologicalOrder {
    type Item = OperationId;
    type IntoIter = std::vec::IntoIter<OperationId>;
    fn into_iter(self) -> Self::IntoIter {
        self.order.into_iter()
    }
}

impl<'a> IntoIterator for &'a TopologicalOrder {
    type Item = OperationId;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, OperationId>>;
    fn into_iter(self) -> Self::IntoIter {
        self.order.iter().copied()
    }
}

impl PartialEq<Vec<OperationId>> for TopologicalOrder {
    fn eq(&self, other: &Vec<OperationId>) -> bool {
        self.order == *other
    }
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub kind: OperationType,
    pub base_cost: i64,
}

#[derive(Clone, Debug)]
pub struct ComputationGraph {
    operations: Vec<Operation>,
    tensors: Vec<Tensor>,
    inputs: HashMap<OperationId, Vec<TensorId>>,
    outputs: HashMap<OperationId, Vec<TensorId>>,
    producer: HashMap<TensorId, OperationId>,
    consumers: HashMap<TensorId, Vec<OperationId>>,
    topological_order: OnceCell<TopologicalOrder>,
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

        let inputs = input
            .inputs
            .iter()
            .cloned()
            .enumerate()
            .map(|(op_idx, op_inputs)| (OperationId(op_idx), op_inputs))
            .collect();

        let outputs = input
            .outputs
            .iter()
            .cloned()
            .enumerate()
            .map(|(op_idx, op_outputs)| (OperationId(op_idx), op_outputs))
            .collect();

        let mut producer = HashMap::new();
        for (op_idx, op_outputs) in input.outputs.iter().enumerate() {
            for &tensor_id in op_outputs {
                let prev = producer.insert(tensor_id, OperationId(op_idx));
                assert!(
                    prev.is_none(),
                    "tensor {:?} has multiple producers",
                    tensor_id
                );
            }
        }

        let mut consumers = HashMap::new();
        for (op_id, input_ids) in input.inputs.iter().enumerate() {
            for &input_id in input_ids {
                consumers
                    .entry(input_id)
                    .or_insert(vec![])
                    .push(OperationId(op_id));
            }
        }

        Self {
            operations,
            tensors,
            inputs,
            outputs,
            producer,
            consumers,
            topological_order: OnceCell::new(),
        }
    }

    pub fn inputs(&self) -> &HashMap<OperationId, Vec<TensorId>> {
        &self.inputs
    }

    pub fn input_ids_for(&self, operation_id: OperationId) -> &[TensorId] {
        self.inputs
            .get(&operation_id)
            .expect(format!("{operation_id:?} does not exist").as_str())
    }

    pub fn inputs_for(&self, operation_id: OperationId) -> Vec<Tensor> {
        self.inputs[&operation_id]
            .iter()
            .map(|id| self.tensors[id.0].clone())
            .collect()
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

    pub fn producer(&self) -> &HashMap<TensorId, OperationId> {
        &self.producer
    }

    pub fn producer_id_of(&self, tensor_id: TensorId) -> Option<OperationId> {
        self.producer.get(&tensor_id).copied()
    }

    pub fn producer_of(&self, tensor_id: TensorId) -> Option<Operation> {
        self.producer_id_of(tensor_id)
            .map(|operation_id| self.operations[operation_id.0].clone())
    }

    pub fn consumers(&self) -> &HashMap<TensorId, Vec<OperationId>> {
        &self.consumers
    }

    pub fn consumer_ids_for(&self, tensor_id: TensorId) -> &[OperationId] {
        self.consumers
            .get(&tensor_id)
            .expect(format!("{tensor_id:?} does not exist").as_str())
    }

    pub fn consumers_for(&self, tensor_id: TensorId) -> Vec<Operation> {
        self.consumers[&tensor_id]
            .iter()
            .map(|id| self.operations[id.0].clone())
            .collect()
    }

    pub fn topological_sort(&self) -> &TopologicalOrder {
        self.topological_order.get_or_init(|| {
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
                                    self.consumers
                                        .get(tensor_id)
                                        .map(|v| v.iter())
                                        .unwrap_or_default()
                                })
                                .for_each(|&consumer_id| {
                                    if state[consumer_id.0] == State::Unvisited {
                                        stack.push(consumer_id);
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
            let mut position = vec![0usize; num_ops];
            for (idx, &op_id) in post_order.iter().enumerate() {
                position[op_id.0] = idx;
            }
            TopologicalOrder {
                order: post_order,
                position,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ComputationGraph, OperationId, OperationType, TensorId};
    use crate::input_format::{DeviceParameters, InputFormat};

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
            for &tensor_id in op_outputs {
                if let Some(consumer_ids) = graph.consumers().get(&tensor_id) {
                    for &consumer in consumer_ids {
                        assert!(
                            position[op_idx.0] < position[consumer.0],
                            "dependency violated: {:?} must come before {:?}",
                            op_idx,
                            consumer
                        );
                    }
                }
            }
        }
    }

    // (no nodes, no edges)
    //
    // result: []
    #[test]
    fn topological_sort_empty_graph() {
        let input = make_input(vec![], vec![], 0);
        let graph = ComputationGraph::new(&input);

        assert_eq!(*graph.topological_sort(), vec![]);
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // result: [op0, op1, op2]
    #[test]
    fn topological_sort_linear_chain() {
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t2 = TensorId(2);
        let t3 = TensorId(3);

        let input = make_input(
            vec![vec![t0], vec![t1], vec![t2]],
            vec![vec![t1], vec![t2], vec![t3]],
            4,
        );
        let graph = ComputationGraph::new(&input);

        assert_eq!(
            *graph.topological_sort(),
            vec![OperationId(0), OperationId(1), OperationId(2)]
        );
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // result: any valid topological order
    #[test]
    fn topological_sort_branching_dag() {
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t2 = TensorId(2);
        let t3 = TensorId(3);
        let t4 = TensorId(4);
        let t5 = TensorId(5);

        let input = make_input(
            vec![vec![t0], vec![t1], vec![t2], vec![t3, t4]],
            vec![vec![t1, t2], vec![t3], vec![t4], vec![t5]],
            6,
        );
        let graph = ComputationGraph::new(&input);
        let order = graph.topological_sort();

        assert_eq!(order.len(), 4);
        assert_is_topological_order(&graph, &order);
    }

    // t0 --> [op0] --t2--> [op1] --t3-->
    //
    // t1 --> [op2] --t4--> [op3] --t5-->
    //
    // result: any valid topological order (two disconnected chains)
    #[test]
    fn topological_sort_disconnected_components() {
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t2 = TensorId(2);
        let t3 = TensorId(3);
        let t4 = TensorId(4);
        let t5 = TensorId(5);

        let input = make_input(
            vec![vec![t0], vec![t2], vec![t1], vec![t4]],
            vec![vec![t2], vec![t3], vec![t4], vec![t5]],
            6,
        );
        let graph = ComputationGraph::new(&input);
        let order = graph.topological_sort();

        assert_eq!(order.len(), 4);
        assert_is_topological_order(&graph, &order);
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // sort([]) = []
    #[test]
    fn sort_empty_input() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let order = graph.topological_sort();

        assert_eq!(order.sort([]), vec![]);
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // sort([op2, op1, op0]) = [op0, op1, op2]
    #[test]
    fn sort_reverses_to_topological_order() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let order = graph.topological_sort();

        assert_eq!(
            order.sort([OperationId(2), OperationId(1), OperationId(0)]),
            vec![OperationId(0), OperationId(1), OperationId(2)]
        );
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // sort([op2, op0]) = [op0, op2]  (subset, reverse order)
    #[test]
    fn sort_subset_in_reverse_order() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let order = graph.topological_sort();

        assert_eq!(
            order.sort([OperationId(2), OperationId(0)]),
            vec![OperationId(0), OperationId(2)]
        );
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // sort([op3, op1]) = [op1, op3]  (sink and mid-node, reversed)
    #[test]
    fn sort_subset_of_branching_dag() {
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
        let order = graph.topological_sort();

        let sorted = order.sort([OperationId(3), OperationId(1)]);
        assert_eq!(sorted.len(), 2);
        assert!(
            order.sort([OperationId(3), OperationId(1)])
                == order.sort([OperationId(1), OperationId(3)]),
            "sort must be stable with respect to topological position"
        );
        // op1 must precede op3 in any valid topological order
        let pos_op1 = sorted.iter().position(|&x| x == OperationId(1)).unwrap();
        let pos_op3 = sorted.iter().position(|&x| x == OperationId(3)).unwrap();
        assert!(pos_op1 < pos_op3);
    }
}
