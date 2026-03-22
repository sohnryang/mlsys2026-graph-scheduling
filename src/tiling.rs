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
            for k in range_k {
                let footprint = input_footprint(m, n, k) + m * n * subgraph_output_ids.len() as i64;
                if footprint > device_params.fast_memory_capacity {
                    continue;
                }
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
