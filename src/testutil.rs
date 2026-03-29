use crate::graph::{ComputationGraph, OperationId, OperationType, Subgraph, TensorId};
use crate::input_format::{DeviceParameters, InputFormat};

pub fn make_input(
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

pub fn make_graph(
    widths: Vec<i64>,
    heights: Vec<i64>,
    inputs: Vec<Vec<TensorId>>,
    outputs: Vec<Vec<TensorId>>,
    op_types: Vec<OperationType>,
) -> ComputationGraph {
    let num_ops = inputs.len();
    let num_tensors = widths.len();
    assert_eq!(num_tensors, heights.len());
    ComputationGraph::new(&InputFormat {
        widths,
        heights,
        inputs,
        outputs,
        base_costs: vec![1; num_ops],
        op_types,
        device_parameters: DeviceParameters {
            fast_memory_capacity: 1,
            slow_memory_bandwidth: 1,
            native_granularity: (1, 1),
        },
    })
}

pub fn pointwise_graph(
    inputs: Vec<Vec<TensorId>>,
    outputs: Vec<Vec<TensorId>>,
    num_tensors: usize,
) -> ComputationGraph {
    let num_ops = inputs.len();
    make_graph(
        vec![1; num_tensors],
        vec![1; num_tensors],
        inputs,
        outputs,
        vec![OperationType::Pointwise; num_ops],
    )
}

pub fn subgraph<'a>(
    graph: &'a ComputationGraph,
    ops: impl IntoIterator<Item = usize>,
) -> Subgraph<'a> {
    Subgraph::from_nodes(graph, ops.into_iter().map(OperationId))
}

/// Build a `ComputationGraph` from a list of directed edges between operations.
///
/// Each edge `(src, dst)` gets its own intermediate tensor. Operations with no
/// incoming edges get one external input tensor, and operations with no outgoing
/// edges get one output tensor. All ops are `Pointwise` with unit dimensions.
pub fn graph_from_edges(n: usize, edges: &[(usize, usize)]) -> ComputationGraph {
    use std::collections::{HashMap, HashSet};

    // Collect predecessors and successors for each op.
    let mut preds: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut succs: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(s, d) in edges {
        succs.entry(s).or_default().push(d);
        preds.entry(d).or_default().push(s);
    }

    // Assign tensor IDs:
    //   - One tensor per edge (intermediate)
    //   - One tensor per source node (no predecessors) as external input
    //   - One tensor per sink node (no successors) as final output
    let mut next_tensor = 0usize;
    let mut alloc = || {
        let id = next_tensor;
        next_tensor += 1;
        TensorId(id)
    };

    // For each edge, create a tensor.
    let mut edge_tensor: HashMap<(usize, usize), TensorId> = HashMap::new();
    for &(s, d) in edges {
        edge_tensor.insert((s, d), alloc());
    }

    // Source nodes get an external input tensor.
    let mut source_tensors: HashMap<usize, TensorId> = HashMap::new();
    for op in 0..n {
        if !preds.contains_key(&op) {
            source_tensors.insert(op, alloc());
        }
    }

    // Sink nodes get an output tensor.
    let mut sink_tensors: HashMap<usize, TensorId> = HashMap::new();
    for op in 0..n {
        if !succs.contains_key(&op) {
            sink_tensors.insert(op, alloc());
        }
    }

    let num_tensors = next_tensor;

    // Build inputs/outputs for each operation.
    let mut op_inputs = vec![vec![]; n];
    let mut op_outputs = vec![vec![]; n];

    // Collect all ops that have predecessors or successors to determine ordering.
    let all_ops: HashSet<usize> = (0..n).collect();

    for op in 0..n {
        // Inputs: external source tensor (if any) + tensors from incoming edges.
        if let Some(&t) = source_tensors.get(&op) {
            op_inputs[op].push(t);
        }
        if let Some(pred_ops) = preds.get(&op) {
            for &p in pred_ops {
                op_inputs[op].push(edge_tensor[&(p, op)]);
            }
        }

        // Outputs: tensors for outgoing edges + sink output tensor (if any).
        if let Some(succ_ops) = succs.get(&op) {
            for &s in succ_ops {
                op_outputs[op].push(edge_tensor[&(op, s)]);
            }
        }
        if let Some(&t) = sink_tensors.get(&op) {
            op_outputs[op].push(t);
        }
    }

    let _ = all_ops; // suppress unused warning
    make_input(op_inputs, op_outputs, num_tensors).into()
}

impl From<InputFormat> for ComputationGraph {
    fn from(input: InputFormat) -> Self {
        ComputationGraph::new(&input)
    }
}

pub fn load_input(filename: &str) -> InputFormat {
    let path = format!("{}/tests/fixtures/{filename}", env!("CARGO_MANIFEST_DIR"));
    let file = std::fs::File::open(&path).unwrap_or_else(|_| panic!("failed to open {path}"));
    serde_json::from_reader(std::io::BufReader::new(file))
        .unwrap_or_else(|e| panic!("failed to parse {path}: {e}"))
}
