use fraction::Fraction;

use crate::{
    graph::{ComputationGraph, Partition, Subgraph},
    input_format::DeviceParameters,
    performance_model::subgraph_latency,
    tiling::search_tile_values,
};

/// Schedules each operation independently in topological order using the
/// largest feasible tile per single-op subgraph and no inter-op retention.
///
/// Ops whose tile search fails (infeasible memory constraints) are skipped.
/// Returns the schedule (same shape as the optimized plan for `write_output`)
/// and the sum of per-op latencies.
pub fn naive_schedule<'a>(
    graph: &'a ComputationGraph,
    device_params: &DeviceParameters,
) -> (Vec<Vec<Partition<'a>>>, f64) {
    let mut schedule = Vec::new();
    let mut total_cost = 0.0f64;

    for &op in graph.topological_sort().iter() {
        let subgraph = Subgraph::from_nodes(graph, std::iter::once(op));
        let Ok(tile) = search_tile_values(&subgraph, device_params, &[]) else {
            continue;
        };
        let metrics = subgraph_latency(device_params, &subgraph, tile, &[]);
        let latency: Fraction = metrics
            .values()
            .flat_map(|v| v.iter().map(|m| m.latency()))
            .sum();
        let cost_f64 = f64::try_from(latency).unwrap();
        total_cost += cost_f64;
        schedule.push(vec![Partition {
            subgraph,
            retained_outputs: vec![],
            tile_size: tile,
        }]);
    }

    (schedule, total_cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ComputationGraph;
    use crate::testutil::load_input;

    // t0(128x128) --> [op0(pw, cost=1000)] --> t1(128x128) --> [op1(pw, cost=100)] --> t2(128x128)
    // Device: fast_memory_capacity=35000, slow_memory_bandwidth=10, native_granularity=(128,128)
    // Naive: two separate subgraphs, each at (128,128,1).
    // op0: memory-bound, each output tile loads t0 slice + writes t1 slice.
    // op1: same.
    // Total naive > fused (3276.8 * 2 = 6553.6).
    #[test]
    fn naive_example1_two_ops() {
        let input = load_input("official_example1.json");
        let graph = ComputationGraph::new(&input);
        let (schedule, total) = naive_schedule(&graph, &input.device_parameters);
        assert_eq!(schedule.len(), 2, "two ops → two single-op subgraphs");
        assert!(total > 0.0);
    }
}
