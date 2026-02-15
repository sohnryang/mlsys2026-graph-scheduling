use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub enum OperationType {
    MatMul,
    Pointwise,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputFormat {
    pub widths: Vec<usize>,
    pub heights: Vec<usize>,
    pub inputs: Vec<Vec<usize>>,
    pub outputs: Vec<Vec<usize>>,
    pub base_costs: Vec<u32>,
    pub op_types: Vec<OperationType>,
    pub fast_memory_capacity: usize,
    pub slow_memory_bandwidth: i32,
    pub native_granularity: (usize, usize),
}
