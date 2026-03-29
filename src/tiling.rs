use std::collections::{HashMap, HashSet};

use crate::{
    graph::{OperationType, Subgraph, TensorId},
    input_format::DeviceParameters,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    TiledM,
    TiledN,
    TiledK,
    Full(i64),
}

impl Axis {
    pub fn occupied_size(&self, m: i64, n: i64, k: i64) -> i64 {
        match self {
            Axis::TiledM => m,
            Axis::TiledN => n,
            Axis::TiledK => k,
            Axis::Full(x) => *x,
        }
    }

    pub fn rank(&self) -> i8 {
        match self {
            Axis::Full(_) => 0,
            Axis::TiledM => 1,
            Axis::TiledN => 2,
            Axis::TiledK => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileShape(pub Axis, pub Axis);

impl TileShape {
    pub fn occupied_size(&self, m: i64, n: i64, k: i64) -> i64 {
        self.0.occupied_size(m, n, k) * self.1.occupied_size(m, n, k)
    }
}

pub struct ConstraintTracker {
    parents: HashMap<Axis, Axis>,
}

#[derive(Debug)]
pub enum SearchError {
    Inconsistent,
    NotFound,
}

impl ConstraintTracker {
    pub fn new() -> Self {
        Self {
            parents: HashMap::from([
                (Axis::TiledM, Axis::TiledM),
                (Axis::TiledN, Axis::TiledN),
                (Axis::TiledK, Axis::TiledK),
            ]),
        }
    }

    pub fn find(&mut self, mut axis: Axis) -> Axis {
        let Some(&parent) = self.parents.get(&axis) else {
            return axis;
        };
        match parent {
            Axis::Full(_) => parent,
            _ => {
                while self.parents[&axis] != axis {
                    let grandparent = self.parents[&self.parents[&axis]];
                    self.parents.insert(axis, grandparent);
                    axis = grandparent;
                }
                axis
            }
        }
    }

    pub fn resolve(&mut self, shape: TileShape) -> TileShape {
        TileShape(self.find(shape.0), self.find(shape.1))
    }

    pub fn merge_with(&mut self, mut other: ConstraintTracker) -> Result<(), SearchError> {
        let axes: Vec<Axis> = other.parents.keys().copied().collect();
        for axis in axes {
            let other_root = other.find(axis);
            if self.parents.contains_key(&axis) {
                let self_root = self.find(axis);
                self.add_equality(self_root, other_root)?;
            } else {
                self.parents.insert(axis, other_root);
            }
        }
        Ok(())
    }

    pub fn add_equality(&mut self, lhs: Axis, rhs: Axis) -> Result<(), SearchError> {
        match (lhs, rhs) {
            (Axis::Full(a), Axis::Full(b)) => {
                if a == b {
                    Ok(())
                } else {
                    Err(SearchError::Inconsistent)
                }
            }
            (constant @ Axis::Full(_), variable) | (variable, constant @ Axis::Full(_)) => {
                let variable_parent = self.find(variable);
                match variable_parent {
                    Axis::Full(_) => {
                        if variable_parent == constant {
                            Ok(())
                        } else {
                            Err(SearchError::Inconsistent)
                        }
                    }
                    _ => {
                        self.parents.insert(variable_parent, constant);
                        Ok(())
                    }
                }
            }
            _ => {
                let lhs_parent = self.find(lhs);
                let rhs_parent = self.find(rhs);
                match (lhs_parent, rhs_parent) {
                    (Axis::Full(_), Axis::Full(_)) => {
                        if lhs_parent == rhs_parent {
                            Ok(())
                        } else {
                            Err(SearchError::Inconsistent)
                        }
                    }
                    (constant @ Axis::Full(_), variable)
                    | (variable, constant @ Axis::Full(_))
                    | (constant, variable) => {
                        self.parents.insert(variable, constant);
                        Ok(())
                    }
                }
            }
        }
    }
}

pub fn propagate_tile_shape(
    subgraph: &Subgraph<'_>,
    start_tensor_id: TensorId,
) -> Result<(HashMap<TensorId, TileShape>, ConstraintTracker), SearchError> {
    let subgraph_inputs = subgraph.input_tensor_ids();
    let graph = subgraph.parent();

    let mut shapes: HashMap<TensorId, TileShape> = HashMap::new();
    let mut visited = HashSet::new();
    let mut worklist = vec![start_tensor_id];
    let mut constraints = ConstraintTracker::new();
    let assign_shape = |shapes: &mut HashMap<TensorId, TileShape>,
                        constraints: &mut ConstraintTracker,
                        tensor_id,
                        shape: TileShape| {
        if let Some(&existing_shape) = shapes.get(&tensor_id) {
            constraints.add_equality(existing_shape.0, shape.0)?;
            constraints.add_equality(existing_shape.1, shape.1)
        } else {
            shapes.insert(tensor_id, shape);
            Ok(())
        }
    };

    assign_shape(
        &mut shapes,
        &mut constraints,
        start_tensor_id,
        TileShape(Axis::TiledM, Axis::TiledN),
    )?;
    while let Some(tensor_id) = worklist.pop() {
        if visited.contains(&tensor_id) {
            continue;
        }
        visited.insert(tensor_id);
        let current_shape = shapes[&tensor_id];

        if subgraph_inputs.contains(&tensor_id) {
            continue;
        }
        let producer_id = graph
            .producer_id_of(tensor_id)
            .ok_or(SearchError::Inconsistent)?;
        let producer_op = graph
            .producer_of(tensor_id)
            .ok_or(SearchError::Inconsistent)?;
        let input_ids = graph.input_ids_for(producer_id);

        match producer_op.kind {
            OperationType::Pointwise => {
                for &input_id in input_ids {
                    assign_shape(&mut shapes, &mut constraints, input_id, current_shape)?;
                    worklist.push(input_id);
                }
            }
            OperationType::MatMul => {
                let input0_id = input_ids[0];
                let input1_id = input_ids[1];

                let (shape0, shape1) = if tensor_id == start_tensor_id {
                    (
                        TileShape(Axis::TiledM, Axis::TiledK),
                        TileShape(Axis::TiledK, Axis::TiledN),
                    )
                } else {
                    let input0 = &graph.tensors()[input0_id.0];
                    let input1 = &graph.tensors()[input1_id.0];
                    (
                        TileShape(current_shape.0, Axis::Full(input0.width)),
                        TileShape(Axis::Full(input1.height), current_shape.1),
                    )
                };
                assign_shape(&mut shapes, &mut constraints, input0_id, shape0)?;
                assign_shape(&mut shapes, &mut constraints, input1_id, shape1)?;
                worklist.extend_from_slice(&[input0_id, input1_id]);
            }
        }
    }

    Ok((shapes, constraints))
}

pub fn search_tile_values(
    subgraph: &Subgraph<'_>,
    device_params: &DeviceParameters,
) -> Result<(i64, i64, i64), SearchError> {
    let subgraph_output_ids = subgraph.output_tensor_ids();
    let mut per_output_shapes = HashMap::new();
    let mut merged_constraints = ConstraintTracker::new();
    for &output_id in subgraph_output_ids.iter() {
        let (shapes, constraints) = propagate_tile_shape(subgraph, output_id)?;
        per_output_shapes.insert(output_id, shapes);
        merged_constraints.merge_with(constraints)?;
    }

    let subgraph_input_ids = subgraph.input_tensor_ids();
    let input_footprint = |m, n, k| {
        subgraph_output_ids.iter().fold(0, |acc, &output_id| {
            acc + subgraph_input_ids.iter().fold(0, |acc, &input_id| {
                acc + per_output_shapes[&output_id]
                    .get(&input_id)
                    .map_or(0, |shape| shape.occupied_size(m, n, k))
            })
        })
    };

    fn ceil_div(x: i64, y: i64) -> i64 {
        (x + y - 1) / y
    }
    let output_dimensions = subgraph_output_ids
        .iter()
        .map(|&tensor_id| {
            let tensor = &subgraph.parent().tensors()[tensor_id.0];
            let operation = subgraph.parent().producer_of(tensor_id).unwrap();
            let reduction_dimension = match operation.kind {
                OperationType::MatMul => {
                    let operation_id = subgraph.parent().producer_id_of(tensor_id).unwrap();
                    subgraph.parent().inputs_for(operation_id)[0].width
                }
                OperationType::Pointwise => 1,
            };
            (
                tensor_id,
                (tensor.height, tensor.width, reduction_dimension),
            )
        })
        .collect::<HashMap<_, _>>();
    let input_traffic = |m, n, k| {
        subgraph_output_ids.iter().fold(0, |acc, &output_id| {
            let (height, width, reduction_dimension) = output_dimensions[&output_id];
            acc + ceil_div(height, m)
                * ceil_div(width, n)
                * ceil_div(reduction_dimension, k)
                * subgraph_input_ids.iter().fold(0, |acc, &input_id| {
                    acc + per_output_shapes[&output_id]
                        .get(&input_id)
                        .map_or(0, |shape| shape.occupied_size(m, n, k))
                })
        })
    };
    let (max_m_value, max_n_value) = device_params.native_granularity;
    let max_k_value = output_dimensions
        .values()
        .map(|(_, _, k)| *k)
        .max()
        .unwrap();
    let clipped_singleton_range = |x, limit| {
        if x <= limit {
            Ok(x..=x)
        } else {
            Err(SearchError::NotFound)
        }
    };
    let range_m = match merged_constraints.find(Axis::TiledM) {
        Axis::Full(x) => clipped_singleton_range(x, max_m_value)?,
        _ => 1..=max_m_value,
    };
    let mut min_traffic = i64::MAX;
    let mut best_values = None;
    let is_spatial_axis_consistent = |axis, x| {
        let mut divided_count = output_dimensions.values().map(|dim| match axis {
            Axis::TiledM => ceil_div(dim.0, x),
            Axis::TiledN => ceil_div(dim.1, x),
            _ => unreachable!(),
        });
        let first = divided_count.next();
        divided_count.all(|dim| first == Some(dim))
    };
    for m in range_m {
        if !is_spatial_axis_consistent(Axis::TiledM, m) {
            continue;
        }
        let range_n = match merged_constraints.find(Axis::TiledN) {
            Axis::Full(x) => clipped_singleton_range(x, max_n_value)?,
            Axis::TiledM => {
                if m <= max_n_value {
                    m..=m
                } else {
                    break;
                }
            }
            _ => 1..=max_n_value,
        };
        for n in range_n {
            if !is_spatial_axis_consistent(Axis::TiledN, n) {
                continue;
            }
            let range_k = match merged_constraints.find(Axis::TiledK) {
                Axis::Full(x) => x..=x,
                Axis::TiledM => m..=m,
                Axis::TiledN => n..=n,
                _ => 1..=max_k_value,
            };
            let k_start = *range_k.start();
            let k_end = *range_k.end();
            let output_footprint = m * n * subgraph_output_ids.len() as i64;
            // Footprint is monotonically non-decreasing in k, so binary-search
            // for the largest k that fits in fast memory.
            let capacity = device_params.fast_memory_capacity;
            let mut lo = k_start;
            let mut hi = k_end + 1;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if input_footprint(m, n, mid) + output_footprint <= capacity {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            // lo == hi == first k that doesn't fit, so iterate k_start..lo.
            for k in k_start..lo {
                let traffic = input_traffic(m, n, k);
                if min_traffic >= traffic {
                    min_traffic = traffic;
                    best_values = Some((m, n, k));
                }
            }
        }
    }
    if let Some(values) = best_values {
        Ok(values)
    } else {
        Err(SearchError::NotFound)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{Axis, TileShape, propagate_tile_shape, search_tile_values};
    use crate::graph::{ComputationGraph, OperationId, OperationType, Subgraph, TensorId};
    use crate::input_format::{DeviceParameters, InputFormat};
    use crate::testutil::{load_input, make_graph, pointwise_graph};

    fn assert_shape(
        shapes: &HashMap<TensorId, TileShape>,
        constraints: &mut super::ConstraintTracker,
        tensor: usize,
        expected: TileShape,
    ) {
        let actual = constraints.resolve(shapes[&TensorId(tensor)]);
        let expected = constraints.resolve(expected);
        assert_eq!(actual, expected, "tensor t{tensor} shape mismatch");
    }

    // t0 --> [op0(pw)] --> t1
    //
    // subgraph: {op0}
    // t0 is input, t1 is output
    // expected: t1=(M,N), t0=(M,N)
    #[test]
    fn single_pointwise() {
        let graph = pointwise_graph(vec![vec![TensorId(0)]], vec![vec![TensorId(1)]], 2);
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(1)).unwrap();

        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
    }

    // t0 --> [op0(pw)] --> t1 --> [op1(pw)] --> t2
    //
    // subgraph: {op0, op1}
    // t0 is input, t2 is output
    // expected: all tensors get (M,N)
    #[test]
    fn chain_of_pointwise() {
        let graph = pointwise_graph(
            vec![vec![TensorId(0)], vec![TensorId(1)]],
            vec![vec![TensorId(1)], vec![TensorId(2)]],
            3,
        );
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(2)).unwrap();

        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
    }

    // t0 --\
    //       +--> [op0(matmul)] --> t2
    // t1 --/
    //
    // subgraph: {op0}
    // t0, t1 are inputs, t2 is output (and subgraph output)
    // expected: t2=(M,N), t0=(M,K), t1=(K,N)
    #[test]
    fn single_matmul_at_output() {
        let graph = make_graph(
            vec![10, 20, 20], // widths:  t0=40x10, t1=10x20, t2=40x20
            vec![40, 10, 40], // heights: t0.w == t1.h == K=10
            vec![vec![TensorId(0), TensorId(1)]],
            vec![vec![TensorId(2)]],
            vec![OperationType::MatMul],
        );
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(2)).unwrap();

        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::TiledK, Axis::TiledN),
        );
    }

    // t0 --\
    //       +--> [op0(matmul)] --> t2 --> [op1(pw)] --> t3
    // t1 --/
    //
    // subgraph: {op0, op1}
    // t0, t1 are inputs, t3 is output
    // t2 is intermediate (matmul output, but NOT subgraph output)
    // expected: t3=(M,N), t2=(M,N), t0=(M,10), t1=(10,N)
    #[test]
    fn matmul_followed_by_pointwise() {
        let graph = make_graph(
            vec![10, 20, 20, 20], // widths:  t0=40x10, t1=10x20, t2=40x20, t3=40x20
            vec![40, 10, 40, 40], // heights: t0.w == t1.h == K=10
            vec![vec![TensorId(0), TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(2)], vec![TensorId(3)]],
            vec![OperationType::MatMul, OperationType::Pointwise],
        );
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(3)).unwrap();

        assert_shape(
            &shapes,
            &mut constraints,
            3,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        // matmul not at subgraph output: input0 gets (current.0, Full(width0)), input1 gets (Full(height1), current.1)
        // current_shape of t2 = (M,N), so t0=(M, Full(10)), t1=(Full(10), N)
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::Full(10)),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::Full(10), Axis::TiledN),
        );
    }

    // t0 --> [op0(pw)] --> t1 --> [op1(pw)] --> t2 --> [op2(pw)] --> t3
    //
    // subgraph: {op0, op1, op2}
    // t0 is input, t3 is output
    // expected: all tensors get (M,N) — pointwise propagates shape unchanged
    #[test]
    fn long_pointwise_chain() {
        let graph = pointwise_graph(
            vec![vec![TensorId(0)], vec![TensorId(1)], vec![TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(2)], vec![TensorId(3)]],
            4,
        );
        let subgraph =
            Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1), OperationId(2)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(3)).unwrap();

        for t in 0..4 {
            assert_shape(
                &shapes,
                &mut constraints,
                t,
                TileShape(Axis::TiledM, Axis::TiledN),
            );
        }
    }

    //                /--t1--> [op1(pw)] --> t3--\
    // t0 --> [op0(pw)]                          +--> [op3(pw)] --> t5
    //                \--t2--> [op2(pw)] --> t4--/
    //
    // subgraph: {op0, op1, op2, op3}
    // t0 is input, t5 is output
    // expected: all tensors get (M,N) — pointwise everywhere
    #[test]
    fn diamond_pointwise() {
        let graph = pointwise_graph(
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
        let subgraph = Subgraph::from_nodes(
            &graph,
            [
                OperationId(0),
                OperationId(1),
                OperationId(2),
                OperationId(3),
            ],
        );
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(5)).unwrap();

        for t in 0..6 {
            assert_shape(
                &shapes,
                &mut constraints,
                t,
                TileShape(Axis::TiledM, Axis::TiledN),
            );
        }
    }

    // t0 --\
    //       +--> [op0(matmul)] --> t2(w=5,h=7) --\
    // t1 --/                                      +--> [op1(matmul)] --> t4
    //                                       t3 --/
    //
    // subgraph: {op0, op1}
    // t0, t1, t3 are inputs, t4 is output
    // op1 output t4 is subgraph output → t2=(M,K), t3=(K,N)
    // op0 output t2 is NOT subgraph output → t0=(M, Full(w0)), t1=(Full(h1), K)
    //   where current_shape of t2 = (M,K), w0=width of t0, h1=height of t1
    #[test]
    fn chained_matmuls() {
        // op0: matmul(t0, t1) -> t2:  t0=4x3, t1=3x5, t2=4x5
        // op1: matmul(t2, t3) -> t4:  t2=4x5, t3=5x8, t4=4x8
        let graph = make_graph(
            vec![3, 5, 5, 8, 8], // widths
            vec![4, 3, 4, 5, 4], // heights
            vec![
                vec![TensorId(0), TensorId(1)], // op0: matmul(t0, t1) -> t2
                vec![TensorId(2), TensorId(3)], // op1: matmul(t2, t3) -> t4
            ],
            vec![vec![TensorId(2)], vec![TensorId(4)]],
            vec![OperationType::MatMul, OperationType::MatMul],
        );
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(4)).unwrap();

        // t4 is output: (M, N)
        assert_shape(
            &shapes,
            &mut constraints,
            4,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        // op1 is at subgraph output: t2=(M,K), t3=(K,N)
        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            3,
            TileShape(Axis::TiledK, Axis::TiledN),
        );
        // op0 is NOT at subgraph output, current_shape of t2=(M,K):
        //   t0=(M, Full(width_of_t0=3)), t1=(Full(height_of_t1=3), K)
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::Full(3)),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::Full(3), Axis::TiledK),
        );
    }

    // t0 --> [op0(pw)] --> t1 --\
    //                            +--> [op1(matmul)] --> t3
    // t2 ---------------------> /
    //
    // subgraph: {op0, op1}
    // t0, t2 are inputs, t3 is output
    // op1 is at subgraph output: t1=(M,K), t2=(K,N)
    // op0 pointwise: t0 gets same shape as t1 = (M,K)
    #[test]
    fn pointwise_feeding_matmul_at_output() {
        // op1: matmul(t1, t2) -> t3:  t1=60x20, t2=20x30, t3=60x30
        // op0: pw(t0) -> t1, so t0 same shape as t1
        let graph = make_graph(
            vec![20, 20, 30, 30], // widths
            vec![60, 60, 20, 60], // heights
            vec![vec![TensorId(0)], vec![TensorId(1), TensorId(2)]],
            vec![vec![TensorId(1)], vec![TensorId(3)]],
            vec![OperationType::Pointwise, OperationType::MatMul],
        );
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(3)).unwrap();

        assert_shape(
            &shapes,
            &mut constraints,
            3,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledK, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
    }

    // t0 --\
    //       +--> [op0(matmul)] --> t2 --> [op1(pw)] --> t3 --\
    // t1 --/                                                  +--> [op2(matmul)] --> t5
    //                                                  t4 --/
    //
    // subgraph: {op0, op1, op2}
    // t0, t1, t4 are inputs, t5 is output
    // op2 at output: t3=(M,K), t4=(K,N)
    // op1 pointwise: t2 gets same shape as t3 = (M,K)
    // op0 NOT at output, current_shape=(M,K): t0=(M, Full(w0)), t1=(Full(h1), K)
    #[test]
    fn matmul_pointwise_matmul_chain() {
        // op0: matmul(t0, t1) -> t2:  t0=4x3, t1=3x5, t2=4x5
        // op1: pw(t2) -> t3, so t3 same shape as t2
        // op2: matmul(t3, t4) -> t5:  t3=4x5, t4=5x11, t5=4x11
        let graph = make_graph(
            vec![3, 5, 5, 5, 11, 11],
            vec![4, 3, 4, 4, 5, 4],
            vec![
                vec![TensorId(0), TensorId(1)],
                vec![TensorId(2)],
                vec![TensorId(3), TensorId(4)],
            ],
            vec![vec![TensorId(2)], vec![TensorId(3)], vec![TensorId(5)]],
            vec![
                OperationType::MatMul,
                OperationType::Pointwise,
                OperationType::MatMul,
            ],
        );
        let subgraph =
            Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1), OperationId(2)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(5)).unwrap();

        assert_shape(
            &shapes,
            &mut constraints,
            5,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            3,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            4,
            TileShape(Axis::TiledK, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::Full(3)),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::Full(3), Axis::TiledK),
        );
    }

    // t0 --> [op0(pw)] --> t1
    //
    // subgraph: {op0} (only op0, but t0 is NOT a subgraph input — it has no producer)
    // This is the minimal single-op subgraph.
    // expected: t1=(M,N), t0=(M,N)
    #[test]
    fn single_op_subgraph() {
        let graph = pointwise_graph(vec![vec![TensorId(0)]], vec![vec![TensorId(1)]], 2);
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(1)).unwrap();

        assert_eq!(shapes.len(), 2);
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
    }

    // t0 --\
    //       +--> [op0(matmul)] --> t2 --\
    // t1 --/                             +--> [op1(pw)] --> t4
    //                              t3 --/
    //
    // subgraph: {op0, op1}
    // t0, t1, t3 are inputs, t4 is output
    // op0 output t2 is NOT subgraph output
    // op1 pointwise: t2 and t3 get (M,N)
    // op0 NOT at output, current_shape of t2 = (M,N):
    //   t0=(M, Full(w0)), t1=(Full(h1), N)
    #[test]
    fn matmul_not_at_output_with_pointwise_consumer() {
        // op0: matmul(t0, t1) -> t2:  t0=3x4, t1=4x5, t2=3x5
        // op1: pw(t2, t3) -> t4
        let graph = make_graph(
            vec![4, 5, 5, 7, 8],
            vec![3, 4, 3, 10, 11],
            vec![
                vec![TensorId(0), TensorId(1)],
                vec![TensorId(2), TensorId(3)],
            ],
            vec![vec![TensorId(2)], vec![TensorId(4)]],
            vec![OperationType::MatMul, OperationType::Pointwise],
        );
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(4)).unwrap();

        assert_shape(
            &shapes,
            &mut constraints,
            4,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            3,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::Full(4)),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::Full(4), Axis::TiledN),
        );
    }

    //                /--t1--\
    // t0 --> [op0(pw)]       +--> [op1(matmul)] --> t3
    //                \--t2--/
    //
    // subgraph: {op0, op1}
    // t0 is input, t3 is output
    // expected: all tensors get (M,M), with constraint M = N = K
    #[test]
    fn diamond_pointwise_to_matmul() {
        // op0: pw(t0) -> (t1, t2): t0=128x128, t1=128x128, t2=128x128
        // op1: matmul(t1, t2) -> t3: t3=128x128
        let graph = make_graph(
            vec![128, 128, 128, 128], // widths
            vec![128, 128, 128, 128], // heights
            vec![
                vec![TensorId(0)],              // op0: pw(t0) -> (t1, t2)
                vec![TensorId(1), TensorId(2)], // op1: matmul(t1, t2) -> t3
            ],
            vec![
                vec![TensorId(1), TensorId(2)], // op0 outputs
                vec![TensorId(3)],              // op1 outputs
            ],
            vec![OperationType::Pointwise, OperationType::MatMul],
        );
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(3)).unwrap();

        // t3 is output: (M, N)
        assert_shape(
            &shapes,
            &mut constraints,
            3,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        // op1 at subgraph output: t1=(M, K), t2=(K, N)
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledK, Axis::TiledN),
        );
        // Constraints unify M=K, N=K, so all axes resolve to the same
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::TiledM),
        );
    }

    // t0 --\
    //       +--> [op0(matmul)] --> t2 --\
    // t1 --/                             +--> [op1(matmul)] --> t4 --\
    //                              t3 --/                             +--> [op2(matmul)] -> t6
    //                                                           t5 --/
    //
    // subgraph: {op0, op1, op2}
    // t0, t1, t3, t5 are inputs, t6 is output
    // op2 output t6 is subgraph output → t4=(M,K), t5=(K,N)
    // op1 output t4 is NOT subgraph output → t2=(M, Full(w2)), t3=(Full(h3), K)
    // op0 output t2 is NOT subgraph output → t0=(M, Full(w0)), t1=(Full(h1), Full(w2))
    #[test]
    fn chained_triple_matmuls() {
        // op0: matmul(t0, t1) -> t2:  t0=8x3, t1=3x5, t2=8x5
        // op1: matmul(t2, t3) -> t4:  t2=8x5, t3=5x7, t4=8x7
        // op2: matmul(t4, t5) -> t6:  t4=8x7, t5=7x11, t6=8x11
        let graph = make_graph(
            vec![3, 5, 5, 7, 7, 11, 11], // widths
            vec![8, 3, 8, 5, 8, 7, 8],   // heights
            vec![
                vec![TensorId(0), TensorId(1)], // op0: matmul(t0, t1) -> t2
                vec![TensorId(2), TensorId(3)], // op1: matmul(t2, t3) -> t4
                vec![TensorId(4), TensorId(5)], // op2: matmul(t4, t5) -> t6
            ],
            vec![vec![TensorId(2)], vec![TensorId(4)], vec![TensorId(6)]],
            vec![
                OperationType::MatMul,
                OperationType::MatMul,
                OperationType::MatMul,
            ],
        );
        let subgraph =
            Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1), OperationId(2)]);
        let (shapes, mut constraints) = propagate_tile_shape(&subgraph, TensorId(6)).unwrap();

        // t6 is output: (M, N)
        assert_shape(
            &shapes,
            &mut constraints,
            6,
            TileShape(Axis::TiledM, Axis::TiledN),
        );
        // op2 at subgraph output: t4=(M, K), t5=(K, N)
        assert_shape(
            &shapes,
            &mut constraints,
            4,
            TileShape(Axis::TiledM, Axis::TiledK),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            5,
            TileShape(Axis::TiledK, Axis::TiledN),
        );
        // op1 NOT at output, current_shape of t4=(M, K):
        //   t2=(M, Full(w2=5)), t3=(Full(h3=5), K)
        assert_shape(
            &shapes,
            &mut constraints,
            2,
            TileShape(Axis::TiledM, Axis::Full(5)),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            3,
            TileShape(Axis::Full(5), Axis::TiledK),
        );
        // op0 NOT at output, current_shape of t2=(M, Full(5)):
        //   t0=(M, Full(w0=3)), t1=(Full(h1=3), Full(w2=5))
        assert_shape(
            &shapes,
            &mut constraints,
            0,
            TileShape(Axis::TiledM, Axis::Full(3)),
        );
        assert_shape(
            &shapes,
            &mut constraints,
            1,
            TileShape(Axis::Full(3), Axis::Full(5)),
        );
    }

    // Example 1 from the official repo (PROBLEM.md):
    //
    // t0(128x128) --> [op0(pw, cost=1000)] --> t1(128x128) --> [op1(pw, cost=100)] --> t2(128x128)
    //
    // Device: fast_memory_capacity=35000, slow_memory_bandwidth=10, native_granularity=(128,128)
    // Fuse all ops into one subgraph. Expected tile: (128, 128, 1).
    #[test]
    fn official_repo_example1() {
        let input = load_input("official_example1.json");
        let graph = ComputationGraph::new(&input);
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);

        let (m, n, k) = search_tile_values(&subgraph, &input.device_parameters).unwrap();
        assert_eq!((m, n, k), (128, 128, 1));
    }

    // Example 2 from the official repo (PROBLEM.md):
    //
    // t0(256x256) --> [op0(pw, cost=1000)] --> t1(256x256) --> [op1(pw, cost=100)] --> t2(256x256)
    //
    // Device: fast_memory_capacity=35000, slow_memory_bandwidth=10, native_granularity=(128,128)
    // Fuse all ops into one subgraph. Expected tile: (128, 128, 1).
    #[test]
    fn official_repo_example2() {
        let input = load_input("official_example2.json");
        let graph = ComputationGraph::new(&input);
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);

        let (m, n, k) = search_tile_values(&subgraph, &input.device_parameters).unwrap();
        assert_eq!((m, n, k), (128, 128, 1));
    }

    // Example 5 from the official repo (PROBLEM.md): Chained MatMul (Split-K)
    //
    // t0(128x128) --\
    //                +--> [op0(matmul, cost=2000)] --> t3(128x128) --\
    // t1(128x128) --/                                                +--> [op1(matmul, cost=2000)] --> t4(128x128)
    //                                                 t2(128x128) --/
    //
    // Device: fast_memory_capacity=45000, slow_memory_bandwidth=10, native_granularity=(128,128)
    // Fuse all ops into one subgraph.
    #[test]
    fn official_repo_example5() {
        let input = load_input("official_example5.json");
        let graph = ComputationGraph::new(&input);
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);

        let (m, n, k) = search_tile_values(&subgraph, &input.device_parameters).unwrap();
        assert_eq!((m, n, k), (128, 128, 43));
    }

    // Variant of example 5 with 256x256 tensors.
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
            inputs: vec![vec![TensorId(0), TensorId(1)], vec![TensorId(3), TensorId(2)]],
            outputs: vec![vec![TensorId(3)], vec![TensorId(4)]],
            base_costs: vec![2000, 2000],
            op_types: vec![OperationType::MatMul, OperationType::MatMul],
            device_parameters: device_params.clone(),
        });
        let subgraph = Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]);

        let (m, n, k) = search_tile_values(&subgraph, &device_params).unwrap();
        assert_eq!((m, n, k), (64, 128, 52));
    }
}
