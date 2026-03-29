use std::{
    collections::{HashMap, HashSet},
    iter,
};

use good_lp::{
    Expression, Solution, SolverModel, constraint, solvers::scip::scip, variable, variables,
};

use crate::graph::{ComputationGraph, Subgraph};

pub fn extract_convex_subgraphs(graph: &ComputationGraph) -> HashSet<Subgraph<'_>> {
    let mut execution_states = HashSet::from([Subgraph::from_nodes(graph, iter::empty())]);
    let topological_order = graph.topological_sort();
    let mut stack = vec![Subgraph::from_nodes(graph, iter::empty())];
    while let Some(execution_state) = stack.pop() {
        for &operation_id in topological_order.iter() {
            if execution_state.contains(operation_id) {
                continue;
            }

            let dependency_satisfied = graph
                .input_ids_for(operation_id)
                .iter()
                .filter_map(|&tensor_id| graph.producer_id_of(tensor_id))
                .all(|producer_id| execution_state.contains(producer_id));
            if !dependency_satisfied {
                continue;
            }

            let mut next_executed_subgraph = execution_state.clone();
            next_executed_subgraph.insert(operation_id);
            if execution_states.contains(&next_executed_subgraph) {
                continue;
            }

            execution_states.insert(next_executed_subgraph.clone());
            stack.push(next_executed_subgraph);
        }
    }

    let mut convex_subgraphs = HashSet::new();
    for state0 in execution_states.iter() {
        for state1 in execution_states.iter() {
            if !state0.is_subset(state1) {
                continue;
            }
            let convex_subgraph = state1.subtract(state0);
            if convex_subgraph.components() == 1 {
                convex_subgraphs.insert(convex_subgraph);
            }
        }
    }
    convex_subgraphs
}

pub fn optimize_execution_plan<'a>(
    graph: &'a ComputationGraph,
    costs: &[(Subgraph<'a>, f64)],
) -> HashSet<Subgraph<'a>> {
    let mut vars = variables!();
    let u_vars = (0..costs.len())
        .map(|_| vars.add(variable().binary()))
        .collect::<Vec<_>>();
    let objective = u_vars
        .iter()
        .zip(costs.iter().map(|c| c.1))
        .map(|(&u, c)| c * u)
        .sum::<Expression>();
    let mut model = vars.minimise(objective).using(scip);

    let mut operation_to_subgraph_indices = HashMap::new();
    for (i, (subgraph, _)) in costs.iter().enumerate() {
        for node in subgraph
            .input_tensor_ids()
            .iter()
            .filter_map(|&tensor_id| graph.producer_id_of(tensor_id))
        {
            operation_to_subgraph_indices
                .entry(node)
                .or_insert(vec![])
                .push(i);
        }
    }

    for operation_id in graph.topological_sort() {
        let operation_cover_count = u_vars
            .iter()
            .enumerate()
            .filter_map(|(i, &u)| {
                if costs[i].0.contains(operation_id) {
                    Some(Expression::from(u))
                } else {
                    None
                }
            })
            .sum::<Expression>();
        model = operation_to_subgraph_indices
            .get(&operation_id)
            .unwrap_or(&vec![])
            .iter()
            .fold(model, |model, &subgraph_idx| {
                model.with(constraint!(
                    operation_cover_count.clone() >= u_vars[subgraph_idx]
                ))
            });
        model = model.with(constraint!(operation_cover_count >= 1));
    }

    let solution = model.solve().unwrap();
    u_vars
        .iter()
        .zip(costs)
        .filter_map(|(&u, (subgraph, _))| {
            if solution.value(u) > 0.5 {
                Some(subgraph.clone())
            } else {
                None
            }
        })
        .collect::<HashSet<_>>()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::extract_convex_subgraphs;
    use crate::{
        graph::{ComputationGraph, OperationId, Subgraph, TensorId},
        schedule::optimize_execution_plan,
        testutil::make_input,
    };

    //              /--t1--> [op1] --t3--\
    // t0 --> [op0]                       +--> [op3] --t5-->
    //              \--t2--> [op2] --t4--/
    //
    // convex subgraphs and costs: {
    //   {op0} -> 20,
    //   {op1} -> 30,
    //   {op2} -> 25,
    //   {op3} -> 40,
    //   {op0, op1} -> 45,
    //   {op0, op2} -> 42,
    //   {op1, op3} -> 55,
    //   {op2, op3} -> 52,
    //   {op0, op1, op2} -> 65,
    //   {op1, op2, op3} -> 70,
    //   {op0, op1, op2, op3} -> 80,
    // }
    #[test]
    fn skip_connection_convex_subgraphs() {
        let input = make_input(
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
        let graph = ComputationGraph::new(&input);
        let convex_subgraphs = extract_convex_subgraphs(&graph);
        let costs: [(_, f64); _] = [
            (Subgraph::from_nodes(&graph, [OperationId(0)]), 20.0),
            (Subgraph::from_nodes(&graph, [OperationId(1)]), 30.0),
            (Subgraph::from_nodes(&graph, [OperationId(2)]), 25.0),
            (Subgraph::from_nodes(&graph, [OperationId(3)]), 40.0),
            (
                Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1)]),
                45.0,
            ),
            (
                Subgraph::from_nodes(&graph, [OperationId(0), OperationId(2)]),
                42.0,
            ),
            (
                Subgraph::from_nodes(&graph, [OperationId(1), OperationId(3)]),
                55.0,
            ),
            (
                Subgraph::from_nodes(&graph, [OperationId(2), OperationId(3)]),
                52.0,
            ),
            (
                Subgraph::from_nodes(&graph, [OperationId(0), OperationId(1), OperationId(2)]),
                65.0,
            ),
            (
                Subgraph::from_nodes(&graph, [OperationId(1), OperationId(2), OperationId(3)]),
                70.0,
            ),
            (
                Subgraph::from_nodes(
                    &graph,
                    [
                        OperationId(0),
                        OperationId(1),
                        OperationId(2),
                        OperationId(3),
                    ],
                ),
                80.0,
            ),
        ];
        let expected = HashSet::from_iter(costs.iter().map(|(s, _)| s.clone()));
        assert_eq!(convex_subgraphs, expected);

        let selected_subgraphs = optimize_execution_plan(&graph, &costs);
        dbg!(
            selected_subgraphs
                .iter()
                .map(|s| s.nodes())
                .collect::<Vec<_>>()
        );
    }
}
