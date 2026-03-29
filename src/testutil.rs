use crate::graph::{ComputationGraph, OperationType, TensorId};
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

pub fn load_input(filename: &str) -> InputFormat {
    let path = format!("{}/tests/fixtures/{filename}", env!("CARGO_MANIFEST_DIR"));
    let file = std::fs::File::open(&path).unwrap_or_else(|_| panic!("failed to open {path}"));
    serde_json::from_reader(std::io::BufReader::new(file))
        .unwrap_or_else(|e| panic!("failed to parse {path}: {e}"))
}
