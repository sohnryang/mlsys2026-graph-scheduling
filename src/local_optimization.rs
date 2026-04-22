use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};

use fraction::Fraction;

use crate::{
    graph::{OperationId, Partition, Subgraph, TensorId},
    input_format::DeviceParameters,
    performance_model::subgraph_latency,
    tiling::search_tile_values,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct State<'a> {
    latency: Fraction,
    executed_partition: Partition<'a>,
}

impl Ord for State<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.latency.cmp(&self.latency)
    }
}

impl PartialOrd for State<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Lazily enumerates subsets of live boundary tensors that fit in fast memory.
/// Tensors already in `retained` that are still live are pre-selected (free to
/// keep); only subsets of the remaining live tensors are enumerated against the
/// leftover capacity.
struct RetentionCandidates {
    candidates: Vec<(TensorId, i64)>,
    /// Stack frame: (selected tensor ids, next candidate index, remaining capacity).
    stack: Vec<(Vec<TensorId>, usize, i64)>,
}

impl RetentionCandidates {
    fn new(candidates: Vec<(TensorId, i64)>, fast_mem_cap: i64, reserved_size: i64) -> Self {
        let remaining = fast_mem_cap - reserved_size;
        Self {
            candidates,
            stack: if remaining >= 0 {
                vec![(vec![], 0, remaining)]
            } else {
                vec![]
            },
        }
    }
}

impl Iterator for RetentionCandidates {
    type Item = Vec<TensorId>;

    fn next(&mut self) -> Option<Self::Item> {
        let (selected, next_idx, remaining) = self.stack.pop()?;

        // Push children in reverse order so the smallest index is popped first.
        for i in (next_idx..self.candidates.len()).rev() {
            let (tid, size) = self.candidates[i];
            if size <= remaining {
                let mut child = selected.clone();
                child.push(tid);
                self.stack.push((child, i + 1, remaining - size));
            }
        }

        Some(selected)
    }
}

const MAX_STAGE_SIZE: usize = 3;

/// Enumerates non-empty connected subsets `S` of `subgraph \ executed` such
/// that `executed ∪ S` is a downset (closed under predecessors taken within
/// `subgraph`). Growth starts from the frontier — ops in `subgraph \ executed`
/// whose in-subgraph predecessors are all in `executed` — and extends by any
/// node whose preds are all in `executed ∪ S` (preserving the downset) and
/// which is adjacent to `S` (preserving connectivity). Adjacency is undirected:
/// producer↔consumer or co-consumer. Stages are capped at `MAX_STAGE_SIZE`.
// TODO: avoid materializing all the stages and make this some kind of an iterator.
fn enum_next_stages<'a>(subgraph: &Subgraph<'a>, executed: &Subgraph<'a>) -> Vec<Subgraph<'a>> {
    let graph = subgraph.parent();
    let nodes = subgraph.nodes();

    // Pre-compute in-subgraph predecessors and undirected neighbors per node.
    let mut preds: HashMap<OperationId, Vec<OperationId>> = HashMap::new();
    let mut neighbors: HashMap<OperationId, HashSet<OperationId>> = HashMap::new();
    for &op in nodes {
        let p: Vec<OperationId> = graph
            .input_ids_for(op)
            .iter()
            .filter_map(|&t| graph.producer_id_of(t))
            .filter(|&q| subgraph.contains(q))
            .collect();
        preds.insert(op, p);

        let mut nbrs: HashSet<OperationId> = HashSet::new();
        for &t in graph.output_ids_for(op) {
            if let Some(consumers) = graph.consumers().get(&t) {
                for &c in consumers {
                    if subgraph.contains(c) && c != op {
                        nbrs.insert(c);
                    }
                }
            }
        }
        for &t in graph.input_ids_for(op) {
            if let Some(producer) = graph.producer_id_of(t) {
                if subgraph.contains(producer) && producer != op {
                    nbrs.insert(producer);
                }
            }
            if let Some(consumers) = graph.consumers().get(&t) {
                for &c in consumers {
                    if subgraph.contains(c) && c != op {
                        nbrs.insert(c);
                    }
                }
            }
        }
        neighbors.insert(op, nbrs);
    }

    // frontier := { v ∈ V \ D : preds[v] ⊆ D }
    let frontier: Vec<OperationId> = nodes
        .iter()
        .copied()
        .filter(|&op| !executed.contains(op))
        .filter(|&op| preds[&op].iter().all(|&p| executed.contains(p)))
        .collect();

    // Iterative GROW: DFS over connected downset-preserving extensions.
    let mut seen: HashSet<BTreeSet<OperationId>> = HashSet::new();
    let mut stack: Vec<BTreeSet<OperationId>> = frontier
        .iter()
        .map(|&seed| {
            let mut s = BTreeSet::new();
            s.insert(seed);
            s
        })
        .collect();

    while let Some(s) = stack.pop() {
        if !seen.insert(s.clone()) {
            continue;
        }
        if s.len() >= MAX_STAGE_SIZE {
            continue;
        }
        for &v in nodes {
            if executed.contains(v) || s.contains(&v) {
                continue;
            }
            // v ready: preds[v] ⊆ D ∪ S
            if !preds[&v]
                .iter()
                .all(|&p| executed.contains(p) || s.contains(&p))
            {
                continue;
            }
            // v connected: adj[v] ∩ S ≠ ∅
            if !neighbors[&v].iter().any(|n| s.contains(n)) {
                continue;
            }
            let mut new_s = s.clone();
            new_s.insert(v);
            stack.push(new_s);
        }
    }

    seen.into_iter()
        .map(|s| Subgraph::from_nodes(graph, s.into_iter()))
        .collect()
}

pub fn partition_subgraph<'a>(
    subgraph: &Subgraph<'a>,
    device_params: &DeviceParameters,
) -> Option<Vec<(Partition<'a>, Fraction)>> {
    let graph = subgraph.parent();
    let subgraph_outputs = subgraph.output_tensor_ids();
    let empty_partition = Partition {
        subgraph: Subgraph::from_nodes(graph, std::iter::empty()),
        retained_outputs: vec![],
        tile_size: (-1, -1, -1),
    };
    let mut costs = HashMap::from([(empty_partition.clone(), Fraction::from(0i64))]);
    let mut parents = HashMap::new();
    let mut heap = BinaryHeap::from([State {
        latency: 0i64.into(),
        executed_partition: empty_partition,
    }]);
    let mut best_latency = None;
    let mut best_partition = None;
    while let Some(State {
        latency,
        executed_partition,
    }) = heap.pop()
    {
        if executed_partition.subgraph.nodes() == subgraph.nodes() {
            if latency < best_latency.unwrap_or(i64::MAX.into()) {
                best_latency = Some(latency);
                best_partition = Some(executed_partition);
            }
            continue;
        }
        if latency >= best_latency.unwrap_or(i64::MAX.into())
            || latency > costs[&executed_partition]
        {
            continue;
        }

        for stage in enum_next_stages(subgraph, &executed_partition.subgraph) {
            // Build new executed subgraph = executed ∪ stage.
            let mut new_executed = executed_partition.subgraph.clone();
            for &op in stage.nodes() {
                new_executed.insert(op);
            }

            // Retention candidates: tensors produced by the current stage
            // that have consumers outside the stage (still needed later).
            let stage_live_tensors: Vec<(TensorId, i64)> = stage
                .output_tensor_ids()
                .into_iter()
                .filter_map(|tid| {
                    if !subgraph_outputs.contains(&tid) {
                        Some((tid, graph.tensors()[tid.0].size()))
                    } else {
                        None
                    }
                })
                .collect();

            let reserved_fast_memory_input: i64 = executed_partition
                .retained_outputs
                .iter()
                .map(|&tensor_id| graph.tensors()[tensor_id.0].size())
                .sum();
            for retention in RetentionCandidates::new(
                stage_live_tensors,
                device_params.fast_memory_capacity,
                reserved_fast_memory_input,
            ) {
                let retained_tensor_ids = [
                    executed_partition.retained_outputs.clone(),
                    retention.clone(),
                ]
                .concat();
                let Ok(tile_size) =
                    search_tile_values(subgraph, device_params, &retained_tensor_ids)
                else {
                    continue;
                };
                let stage_performance_metrics =
                    subgraph_latency(device_params, &stage, tile_size, &retained_tensor_ids);
                let new_partition = Partition {
                    subgraph: new_executed.clone(),
                    retained_outputs: retention,
                    tile_size,
                };
                let new_latency = latency
                    + stage_performance_metrics
                        .values()
                        .map(|metrics| {
                            metrics
                                .iter()
                                .map(|metric| metric.latency())
                                .sum::<Fraction>()
                        })
                        .sum::<Fraction>();

                if new_latency >= *costs.get(&new_partition).unwrap_or(&i64::MAX.into()) {
                    continue;
                }

                costs.insert(new_partition.clone(), new_latency);
                parents.insert(new_partition.clone(), executed_partition.clone());
                heap.push(State {
                    latency: new_latency,
                    executed_partition: new_partition,
                });
            }
        }
    }

    let Some(last_partition) = best_partition else {
        return None;
    };
    let mut execution_state_chain = vec![last_partition];
    while let Some(prev_partition) = parents.get(execution_state_chain.last().unwrap()) {
        execution_state_chain.push(prev_partition.clone());
    }
    let partition_chain = execution_state_chain[0..execution_state_chain.len().strict_sub(1)]
        .iter()
        .zip(execution_state_chain[1..execution_state_chain.len()].iter())
        .map(|(after, before)| {
            (
                Partition {
                    subgraph: after.subgraph.subtract(&before.subgraph),
                    retained_outputs: after.retained_outputs.clone(),
                    tile_size: after.tile_size,
                },
                costs[after],
            )
        })
        .rev()
        .collect::<Vec<_>>();
    Some(partition_chain)
}
