use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::input_format::InputFormat;

#[derive(Clone, Copy, Debug, Deserialize)]
pub enum OperationType {
    MatMul,
    Pointwise,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct TensorId(pub usize);

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: TensorId,
    pub width: i64,
    pub height: i64,
}

impl Tensor {
    pub fn size(&self) -> i64 {
        self.width * self.height
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
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
    topological_order: OnceLock<TopologicalOrder>,
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
            topological_order: OnceLock::new(),
        }
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
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
        self.consumers.get(&tensor_id).map_or(&[], |v| v.as_slice())
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

#[derive(Debug, Clone)]
pub struct Subgraph<'a> {
    parent: &'a ComputationGraph,
    nodes: Vec<OperationId>,
}

impl<'a> Subgraph<'a> {
    pub fn from_nodes(
        parent: &'a ComputationGraph,
        iter: impl IntoIterator<Item = OperationId>,
    ) -> Self {
        let nodes = parent.topological_sort().sort(iter);
        Self { parent, nodes }
    }

    pub fn parent(&self) -> &'a ComputationGraph {
        self.parent
    }

    pub fn nodes(&self) -> &[OperationId] {
        &self.nodes
    }

    pub fn contains(&self, operation_id: OperationId) -> bool {
        let topo = self.parent.topological_sort();
        self.nodes
            .binary_search_by_key(&topo.position[operation_id.0], |n| topo.position[n.0])
            .is_ok()
    }

    pub fn is_subset(&self, other: &Subgraph) -> bool {
        let topo = self.parent.topological_sort();
        let mut j = 0;
        for &node in &self.nodes {
            let pos = topo.position[node.0];
            while j < other.nodes.len() && topo.position[other.nodes[j].0] < pos {
                j += 1;
            }
            if j >= other.nodes.len() || other.nodes[j] != node {
                return false;
            }
            j += 1;
        }
        true
    }

    pub fn subtract(&self, other: &Subgraph) -> Subgraph<'a> {
        let topo = self.parent.topological_sort();
        let mut j = 0;
        let mut nodes = Vec::new();
        for &node in &self.nodes {
            let pos = topo.position[node.0];
            while j < other.nodes.len() && topo.position[other.nodes[j].0] < pos {
                j += 1;
            }
            if j < other.nodes.len() && other.nodes[j] == node {
                j += 1;
            } else {
                nodes.push(node);
            }
        }
        Subgraph {
            parent: self.parent,
            nodes,
        }
    }

    pub fn insert(&mut self, operation_id: OperationId) {
        let topo = self.parent.topological_sort();
        let pos = topo.position[operation_id.0];
        let idx = self
            .nodes
            .binary_search_by_key(&pos, |n| topo.position[n.0])
            .unwrap_or_else(|i| i);
        self.nodes.insert(idx, operation_id);
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the set of tensors consumed by nodes in this subgraph that are
    /// not produced by any node in this subgraph.
    pub fn input_tensor_ids(&self) -> HashSet<TensorId> {
        let node_set: HashSet<OperationId> = self.nodes.iter().copied().collect();
        self.nodes
            .iter()
            .flat_map(|&op| self.parent.input_ids_for(op))
            .copied()
            .filter(|&tensor_id| {
                self.parent
                    .producer_id_of(tensor_id)
                    .map_or(true, |producer| !node_set.contains(&producer))
            })
            .collect()
    }

    /// Returns the number of connected components in this subgraph.
    pub fn components(&self) -> usize {
        let node_set: HashSet<OperationId> = self.nodes.iter().copied().collect();
        let mut visited: HashSet<OperationId> = HashSet::new();
        let mut count = 0;

        for &start in &self.nodes {
            if visited.contains(&start) {
                continue;
            }
            count += 1;
            let mut stack = vec![start];
            while let Some(op) = stack.pop() {
                if !visited.insert(op) {
                    continue;
                }
                // Follow output tensors to consumers within the subgraph.
                for &tensor_id in self.parent.output_ids_for(op) {
                    if let Some(consumers) = self.parent.consumers().get(&tensor_id) {
                        for &c in consumers {
                            if node_set.contains(&c) && !visited.contains(&c) {
                                stack.push(c);
                            }
                        }
                    }
                }
                // Follow input tensors back to producers within the subgraph.
                for &tensor_id in self.parent.input_ids_for(op) {
                    if let Some(producer) = self.parent.producer_id_of(tensor_id) {
                        if node_set.contains(&producer) && !visited.contains(&producer) {
                            stack.push(producer);
                        }
                    }
                }
            }
        }

        count
    }

    /// Returns the set of tensors produced by nodes in this subgraph that are
    /// consumed by nodes outside this subgraph, or have no consumers.
    pub fn output_tensor_ids(&self) -> HashSet<TensorId> {
        let node_set: HashSet<OperationId> = self.nodes.iter().copied().collect();
        self.nodes
            .iter()
            .flat_map(|&op| self.parent.output_ids_for(op))
            .copied()
            .filter(|&tensor_id| {
                self.parent
                    .consumers()
                    .get(&tensor_id)
                    .map_or(true, |consumers| {
                        consumers.iter().any(|c| !node_set.contains(c))
                    })
            })
            .collect()
    }
}

impl PartialEq for Subgraph<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
    }
}

impl Eq for Subgraph<'_> {}

impl std::hash::Hash for Subgraph<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.nodes.hash(state);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Partition<'a> {
    pub subgraph: Subgraph<'a>,
    pub retained_outputs: Vec<TensorId>,
    pub tile_size: (i64, i64, i64),
}

#[cfg(test)]
mod tests {
    use super::{ComputationGraph, OperationId, TensorId};
    use crate::testutil::{make_input, subgraph};

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

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0, op1} ⊆ {op0, op1, op2}
    #[test]
    fn is_subset_proper_subset() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let sub = subgraph(&graph, [0, 1]);
        let sup = subgraph(&graph, [0, 1, 2]);

        assert!(sub.is_subset(&sup));
        assert!(!sup.is_subset(&sub));
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0, op1, op2} ⊆ {op0, op1, op2}
    #[test]
    fn is_subset_equal_sets() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let a = subgraph(&graph, [0, 1, 2]);
        let b = subgraph(&graph, [0, 1, 2]);

        assert!(a.is_subset(&b));
        assert!(b.is_subset(&a));
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {} ⊆ {op0, op1}
    #[test]
    fn is_subset_empty_is_subset_of_any() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let empty = subgraph(&graph, []);
        let non_empty = subgraph(&graph, [0, 1]);

        assert!(empty.is_subset(&non_empty));
        assert!(empty.is_subset(&empty));
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // {op1, op3} ⊄ {op0, op2} (disjoint subsets from branches)
    #[test]
    fn is_subset_disjoint_subgraphs() {
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
        let a = subgraph(&graph, [1, 3]);
        let b = subgraph(&graph, [0, 2]);

        assert!(!a.is_subset(&b));
        assert!(!b.is_subset(&a));
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // {op0, op2} ⊆ {op0, op1, op2, op3}
    // {op1} ⊆ {op0, op1, op2, op3}
    #[test]
    fn is_subset_partial_overlap_in_dag() {
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
        let full = subgraph(&graph, [0, 1, 2, 3]);
        let branch = subgraph(&graph, [0, 2]);
        let single = subgraph(&graph, [1]);

        assert!(branch.is_subset(&full));
        assert!(single.is_subset(&full));
        assert!(!full.is_subset(&branch));
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0, op1, op2} - {op1} = {op0, op2}
    #[test]
    fn subtract_single_element() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let all = subgraph(&graph, [0, 1, 2]);
        let mid = subgraph(&graph, [1]);

        let result = all.subtract(&mid);
        assert_eq!(result.nodes(), &[OperationId(0), OperationId(2)]);
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0, op1, op2} - {op0, op1, op2} = {}
    #[test]
    fn subtract_equal_sets_gives_empty() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let a = subgraph(&graph, [0, 1, 2]);
        let b = subgraph(&graph, [0, 1, 2]);

        assert!(a.subtract(&b).nodes().is_empty());
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0, op2} - {} = {op0, op2}
    #[test]
    fn subtract_empty_is_identity() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let a = subgraph(&graph, [0, 2]);
        let empty = subgraph(&graph, []);

        assert_eq!(
            a.subtract(&empty).nodes(),
            &[OperationId(0), OperationId(2)]
        );
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {} - {op0, op1} = {}
    #[test]
    fn subtract_from_empty_gives_empty() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let empty = subgraph(&graph, []);
        let b = subgraph(&graph, [0, 1]);

        assert!(empty.subtract(&b).nodes().is_empty());
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // {op0, op1, op2, op3} - {op1, op3} = {op0, op2}
    #[test]
    fn subtract_disjoint_result_in_dag() {
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
        let full = subgraph(&graph, [0, 1, 2, 3]);
        let branch = subgraph(&graph, [1, 3]);

        let result = full.subtract(&branch);
        assert_eq!(result.nodes(), &[OperationId(0), OperationId(2)]);
    }

    // empty subgraph has 0 components
    #[test]
    fn components_empty_subgraph() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)]],
            vec![vec![TensorId(1)], vec![TensorId(2)]],
            3,
        );
        let graph = ComputationGraph::new(&input);
        let sub = subgraph(&graph, []);

        assert_eq!(sub.components(), 0);
    }

    // t0 --> [op0] --t1-->
    //
    // {op0} => 1 component
    #[test]
    fn components_single_node() {
        let input = make_input(vec![vec![TensorId(0)]], vec![vec![TensorId(1)]], 2);
        let graph = ComputationGraph::new(&input);
        let sub = subgraph(&graph, [0]);

        assert_eq!(sub.components(), 1);
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0, op1, op2} => 1 component (all connected in a chain)
    #[test]
    fn components_linear_chain() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let sub = subgraph(&graph, [0, 1, 2]);

        assert_eq!(sub.components(), 1);
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0, op2} => 2 components (op1 removed breaks the chain)
    #[test]
    fn components_broken_chain() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let sub = subgraph(&graph, [0, 2]);

        assert_eq!(sub.components(), 2);
    }

    // t0 --> [op0] --t2--> [op1] --t3-->
    //
    // t1 --> [op2] --t4--> [op3] --t5-->
    //
    // {op0, op1, op2, op3} => 2 components (two disconnected chains)
    #[test]
    fn components_two_disconnected_chains() {
        let input = make_input(
            vec![
                vec![TensorId(0)],
                vec![TensorId(2)],
                vec![TensorId(1)],
                vec![TensorId(4)],
            ],
            vec![
                vec![TensorId(2)],
                vec![TensorId(3)],
                vec![TensorId(4)],
                vec![TensorId(5)],
            ],
            6,
        );
        let graph = ComputationGraph::new(&input);
        let sub = subgraph(&graph, [0, 1, 2, 3]);

        assert_eq!(sub.components(), 2);
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // {op0, op1, op2, op3} => 1 component
    #[test]
    fn components_branching_dag_all_nodes() {
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
        let sub = subgraph(&graph, [0, 1, 2, 3]);

        assert_eq!(sub.components(), 1);
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // {op1, op2} => 2 components (branches without shared parent or child)
    #[test]
    fn components_parallel_branches() {
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
        let sub = subgraph(&graph, [1, 2]);

        assert_eq!(sub.components(), 2);
    }

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // {op1, op2, op3} => 1 component (op3 connects both branches)
    #[test]
    fn components_branches_joined_by_sink() {
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
        let sub = subgraph(&graph, [1, 2, 3]);

        assert_eq!(sub.components(), 1);
    }

    // t0 --> [op0] --t1--> [op1] --t2--> [op2] --t3-->
    //
    // {op0} - {op1, op2} = {op0}  (other has elements not in self)
    #[test]
    fn subtract_with_non_overlapping_other() {
        let input = make_input(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let graph = ComputationGraph::new(&input);
        let a = subgraph(&graph, [0]);
        let b = subgraph(&graph, [1, 2]);

        assert_eq!(a.subtract(&b).nodes(), &[OperationId(0)]);
    }

    // t0 ---+--> [op0] --> t2 --+---> [op1] --> t4
    // t1 --/                     \
    //                             +--> [op2] --> t5
    //                       t3 --/
    // output tensor ids of subgraph {op0, op1} : {t2, t4}
    #[test]
    fn output_tensors_of_branching_subgraph() {
        let input = make_input(
            vec![
                vec![TensorId(0), TensorId(1)], // op0 inputs
                vec![TensorId(2)],              // op1 inputs
                vec![TensorId(2), TensorId(3)], // op2 inputs
            ],
            vec![
                vec![TensorId(2)], // op0 outputs
                vec![TensorId(4)], // op1 outputs
                vec![TensorId(5)], // op2 outputs
            ],
            6,
        );
        let graph = ComputationGraph::new(&input);
        let sg = subgraph(&graph, [0, 1]);

        let output_tensors = sg.output_tensor_ids();
        assert_eq!(output_tensors.len(), 2);
        assert!(output_tensors.contains(&TensorId(2)));
        assert!(output_tensors.contains(&TensorId(4)));
    }
}
