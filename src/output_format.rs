use serde::Serialize;

use crate::graph::{OperationId, TensorId};

#[derive(Clone, Debug, Serialize)]
pub struct OutputFormat {
    pub subgraphs: Vec<Vec<OperationId>>,
    pub granularities: Vec<(i64, i64, i64)>,
    pub tensors_to_retain: Vec<Vec<TensorId>>,
    pub traversal_orders: Vec<Option<Vec<i64>>>,
    pub subgraph_latencies: Vec<f64>,
}
