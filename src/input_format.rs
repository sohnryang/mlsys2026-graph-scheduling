use serde::Deserialize;

use crate::graph::{OperationType, TensorId};

#[derive(Clone, Debug, Deserialize)]
pub struct InputFormat {
    pub widths: Vec<i64>,
    pub heights: Vec<i64>,
    pub inputs: Vec<Vec<TensorId>>,
    pub outputs: Vec<Vec<TensorId>>,
    pub base_costs: Vec<i64>,
    pub op_types: Vec<OperationType>,
    #[serde(flatten)]
    pub device_parameters: DeviceParameters,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DeviceParameters {
    pub fast_memory_capacity: i64,
    pub slow_memory_bandwidth: i64,
    pub native_granularity: (i64, i64),
}
