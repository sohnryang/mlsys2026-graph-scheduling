use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use fraction::Fraction;
use rayon::prelude::*;

pub mod global_optimization;
pub mod graph;
pub mod input_format;
pub mod local_optimization;
pub mod naive_scheduler;
pub mod output_format;
pub mod partition;
pub mod performance_model;
#[cfg(test)]
pub mod testutil;
pub mod tiling;
use crate::global_optimization::{extract_convex_subgraphs, optimize_execution_plan};
use crate::graph::{ComputationGraph, Partition, Subgraph, TensorId};
use crate::input_format::{DeviceParameters, InputFormat};
use crate::naive_scheduler::naive_schedule;
use crate::output_format::OutputFormat;
use crate::partition::search_partition;
use crate::performance_model::{subgraph_latency, total_latency};
use crate::tiling::{ceil_div, search_tile_values};

#[derive(Parser)]
struct Cli {
    input_file: PathBuf,
    output_file: PathBuf,
    #[arg(short, long)]
    verbose: bool,
    /// Run the naive single-op baseline and print a latency comparison to stderr.
    #[arg(long)]
    compare_naive: bool,
}

fn main() {
    let cli = Cli::parse();
    let file = File::open(&cli.input_file).expect("failed to open input file");
    let reader = BufReader::new(file);
    let input: InputFormat =
        serde_json::from_reader(reader).expect("failed to parse input file as JSON");

    let graph = ComputationGraph::new(&input);
    let start = Instant::now();
    let convex_subgraphs = extract_convex_subgraphs(&graph)
        .into_iter()
        .collect::<Vec<_>>();
    if cli.verbose {
        eprintln!("convex subgraph extraction took: {:?}", start.elapsed());
        eprintln!("discovered {} convex subgraphs", convex_subgraphs.len());
    }
    let start = Instant::now();
    let subgraph_costs = convex_subgraphs
        .into_par_iter()
        .filter_map(|subgraph| {
            let tile_size = search_tile_values(&subgraph, &input.device_parameters, &[]).ok()?;
            let metrics = subgraph_latency(&input.device_parameters, &subgraph, tile_size, &[]);
            let cost = f64::try_from(total_latency(&metrics)).unwrap();
            let whole_partition = vec![Partition {
                subgraph: subgraph.clone(),
                retained_outputs: vec![],
                tile_size,
            }];
            Some((subgraph, cost, whole_partition))
        })
        .collect::<Vec<_>>();
    if cli.verbose {
        eprintln!("whole-subgraph cost model took: {:?}", start.elapsed());
        eprintln!("subgraph count: {}", subgraph_costs.len());
    }
    let start = Instant::now();
    let execution_plan = optimize_execution_plan(&graph, &subgraph_costs);
    if cli.verbose {
        eprintln!("subgraph selection took: {:?}", start.elapsed());
    }

    let start = Instant::now();
    let execution_plan = execution_plan
        .into_par_iter()
        .map(|(subgraph, _)| {
            let (partitions, _) = search_partition(&subgraph, &input.device_parameters)
                .expect("selected subgraph failed to partition");
            (subgraph, partitions)
        })
        .collect::<Vec<_>>();
    if cli.verbose {
        eprintln!("selected subgraph partitioning took: {:?}", start.elapsed());
    }

    let ordered_plan = topological_sort_subgraphs(execution_plan);
    let optimized_total = write_output(&cli.output_file, &input.device_parameters, ordered_plan);

    if cli.compare_naive {
        let (_, naive_total) = naive_schedule(&graph, &input.device_parameters);
        eprintln!("Naive total latency:     {:.2}", naive_total);
        eprintln!("Optimized total latency: {:.2}", optimized_total);
        if optimized_total > 0.0 {
            eprintln!(
                "Speedup:                 {:.2}x",
                naive_total / optimized_total
            );
        }
    }
}

fn write_output(
    path: &Path,
    device_params: &DeviceParameters,
    ordered_plan: Vec<Vec<Partition<'_>>>,
) -> f64 {
    let mut subgraphs = Vec::new();
    let mut granularities = Vec::new();
    let mut tensors_to_retain = Vec::new();
    let mut traversal_orders = Vec::new();
    let mut subgraph_latencies = Vec::new();

    for partitions in ordered_plan {
        for Partition {
            subgraph: sg,
            retained_outputs: retained,
            tile_size: tile,
        } in partitions
        {
            let metrics = subgraph_latency(device_params, &sg, tile, &retained);
            let latency: Fraction = metrics
                .values()
                .flat_map(|v| v.iter().map(|m| m.latency()))
                .sum();
            let traversal = snake_traversal(&sg, tile);
            subgraphs.push(sg.nodes().to_vec());
            granularities.push(tile);
            tensors_to_retain.push(retained);
            traversal_orders.push(traversal);
            subgraph_latencies.push(f64::try_from(latency).unwrap());
        }
    }

    let total: f64 = subgraph_latencies.iter().sum();
    let output = OutputFormat {
        subgraphs,
        granularities,
        tensors_to_retain,
        traversal_orders,
        subgraph_latencies,
    };
    let file = File::create(path).expect("failed to create output file");
    serde_json::to_writer(BufWriter::new(file), &output).expect("failed to write output");
    total
}

/// Spatial tile order matching `subgraph_latency`'s iteration: snake over the
/// smallest output's `(m_tiles × n_tiles)` grid. Indices are row-major:
/// `m * n_tiles + n`. Returns `None` when the grid has fewer than two rows
/// or columns — snake collapses to raster there, and the output schema
/// expects `null` for the default order.
fn snake_traversal(subgraph: &Subgraph<'_>, tile_size: (i64, i64, i64)) -> Option<Vec<i64>> {
    let graph = subgraph.parent();
    let output_id = subgraph
        .output_tensor_ids()
        .into_iter()
        .min()
        .expect("subgraph must have at least one output");
    let output = &graph.tensors()[output_id.0];
    let (tile_h, tile_w, _) = tile_size;
    let rows = ceil_div(output.height, tile_h);
    let cols = ceil_div(output.width, tile_w);
    if rows < 2 || cols < 2 {
        return None;
    }
    let mut order = Vec::with_capacity((rows * cols) as usize);
    for m in 0..rows {
        for step in 0..cols {
            let n = if m % 2 == 0 { step } else { cols - 1 - step };
            order.push(m * cols + n);
        }
    }
    Some(order)
}

/// Orders selected subgraphs so that if subgraph `B` produces a tensor that
/// subgraph `A` consumes, `B` precedes `A`. Uses Kahn's algorithm over the
/// subgraph-level dependency DAG induced by tensor production/consumption.
/// Returns the partitions of each subgraph in the sorted order.
fn topological_sort_subgraphs<'a>(
    items: Vec<(Subgraph<'a>, Vec<Partition<'a>>)>,
) -> Vec<Vec<Partition<'a>>> {
    let n = items.len();
    let mut producers: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (i, (subgraph, _)) in items.iter().enumerate() {
        for tensor_id in subgraph.output_tensor_ids() {
            producers.entry(tensor_id).or_default().push(i);
        }
    }

    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree = vec![0usize; n];
    for (i, (subgraph, _)) in items.iter().enumerate() {
        let mut added: HashSet<usize> = HashSet::new();
        for tensor_id in subgraph.input_tensor_ids() {
            if let Some(prods) = producers.get(&tensor_id) {
                for &p in prods {
                    if p != i && added.insert(p) {
                        edges[p].push(i);
                        in_degree[i] += 1;
                    }
                }
            }
        }
    }

    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = queue.pop() {
        order.push(i);
        for &j in &edges[i] {
            in_degree[j] -= 1;
            if in_degree[j] == 0 {
                queue.push(j);
            }
        }
    }
    assert_eq!(order.len(), n, "subgraph dependency graph has a cycle");

    let mut partitions: Vec<Option<Vec<Partition<'a>>>> =
        items.into_iter().map(|(_, p)| Some(p)).collect();
    order
        .into_iter()
        .map(|i| partitions[i].take().unwrap())
        .collect()
}
