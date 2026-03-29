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
        schedule::optimize_execution_plan,
        testutil::{graph_from_edges, subgraph},
    };

    // t0 --> [op0] --t1-->
    //
    // optimal: {op0}=100
    #[test]
    fn single_primitive() {
        let graph = graph_from_edges(1, &[]);
        let convex = extract_convex_subgraphs(&graph);
        assert_eq!(convex, HashSet::from([subgraph(&graph, [0])]));

        let costs = [(subgraph(&graph, [0]), 100.0)];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(selected, HashSet::from([subgraph(&graph, [0])]));
    }

    // t0 --> [op0] --t2-->    t1 --> [op1] --t3-->
    //
    // {p0,p1} disconnected → excluded from convex subgraphs
    // optimal: {op0}+{op1}=120
    #[test]
    fn two_independent_outputs() {
        let graph = graph_from_edges(2, &[]);
        let convex = extract_convex_subgraphs(&graph);
        let expected = HashSet::from([subgraph(&graph, [0]), subgraph(&graph, [1])]);
        assert_eq!(convex, expected);

        let costs = [(subgraph(&graph, [0]), 50.0), (subgraph(&graph, [1]), 70.0)];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(selected, expected);
    }

    // t1 --> [op0] --t0--> [op1] --t2-->
    //
    // fusion wins: {op0,op1}=100 < {op0}+{op1}=140
    #[test]
    fn chain2_fusion_wins() {
        let graph = graph_from_edges(2, &[(0, 1)]);
        let costs = [
            (subgraph(&graph, [0]), 80.0),
            (subgraph(&graph, [1]), 60.0),
            (subgraph(&graph, [0, 1]), 100.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(selected, HashSet::from([subgraph(&graph, [0, 1])]));
    }

    // t1 --> [op0] --t0--> [op1] --t2-->
    //
    // singles win: {op0}+{op1}=70 < {op0,op1}=100
    #[test]
    fn chain2_singles_win() {
        let graph = graph_from_edges(2, &[(0, 1)]);
        let costs = [
            (subgraph(&graph, [0]), 30.0),
            (subgraph(&graph, [1]), 40.0),
            (subgraph(&graph, [0, 1]), 100.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0]), subgraph(&graph, [1])])
        );
    }

    //                /--t0--> [op1] --t2--\
    // t4 --> [op0] --+                    +--> [op3] --t5-->
    //                \--t1--> [op2] --t3--/
    //
    // optimal: {op0,op1,op2,op3}=80
    #[test]
    fn diamond_full_fusion() {
        let graph = graph_from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let convex = extract_convex_subgraphs(&graph);
        let expected_convex = HashSet::from([
            subgraph(&graph, [0]),
            subgraph(&graph, [1]),
            subgraph(&graph, [2]),
            subgraph(&graph, [3]),
            subgraph(&graph, [0, 1]),
            subgraph(&graph, [0, 2]),
            subgraph(&graph, [1, 3]),
            subgraph(&graph, [2, 3]),
            subgraph(&graph, [0, 1, 2]),
            subgraph(&graph, [1, 2, 3]),
            subgraph(&graph, [0, 1, 2, 3]),
        ]);
        assert_eq!(convex, expected_convex);

        let costs = [
            (subgraph(&graph, [0]), 20.0),
            (subgraph(&graph, [1]), 30.0),
            (subgraph(&graph, [2]), 25.0),
            (subgraph(&graph, [3]), 40.0),
            (subgraph(&graph, [0, 1]), 45.0),
            (subgraph(&graph, [0, 2]), 42.0),
            (subgraph(&graph, [1, 3]), 55.0),
            (subgraph(&graph, [2, 3]), 52.0),
            (subgraph(&graph, [0, 1, 2]), 65.0),
            (subgraph(&graph, [1, 2, 3]), 70.0),
            (subgraph(&graph, [0, 1, 2, 3]), 80.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(selected, HashSet::from([subgraph(&graph, [0, 1, 2, 3])]));
    }

    // t2 --> [op0] --t0--> [op1] --t1--> [op2] --t3-->
    //
    // tail fusion: {op0}+{op1,op2}=75
    #[test]
    fn chain3_tail_fusion() {
        let graph = graph_from_edges(3, &[(0, 1), (1, 2)]);
        let costs = [
            (subgraph(&graph, [0]), 20.0),
            (subgraph(&graph, [1]), 30.0),
            (subgraph(&graph, [2]), 40.0),
            (subgraph(&graph, [0, 1]), 50.0),
            (subgraph(&graph, [1, 2]), 55.0),
            (subgraph(&graph, [0, 1, 2]), 100.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0]), subgraph(&graph, [1, 2])])
        );
    }

    //                /--t0--> [op1] --t3-->
    // t2 --> [op0] --+
    //                \--t1--> [op2] --t4-->
    //
    // partial fusion: {op0,op1}+{op2}=65
    #[test]
    fn fanout2_partial_fusion() {
        let graph = graph_from_edges(3, &[(0, 1), (0, 2)]);
        let costs = [
            (subgraph(&graph, [0]), 40.0),
            (subgraph(&graph, [1]), 30.0),
            (subgraph(&graph, [2]), 25.0),
            (subgraph(&graph, [0, 1]), 40.0),
            (subgraph(&graph, [0, 2]), 35.0),
            (subgraph(&graph, [0, 1, 2]), 90.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0, 1]), subgraph(&graph, [2])])
        );
    }

    // t4 --> [op0] --t0--> [op1] --t1--> [op2] --t2--> [op3] --t3--> [op4] --t5-->
    //
    // optimal: {op0,op1}+{op2,op3}+{op4}=140
    #[test]
    fn chain5_partial_fusion() {
        let graph = graph_from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let costs = [
            (subgraph(&graph, [0]), 40.0),
            (subgraph(&graph, [1]), 40.0),
            (subgraph(&graph, [2]), 40.0),
            (subgraph(&graph, [3]), 40.0),
            (subgraph(&graph, [4]), 30.0),
            (subgraph(&graph, [0, 1]), 50.0),
            (subgraph(&graph, [1, 2]), 70.0),
            (subgraph(&graph, [2, 3]), 60.0),
            (subgraph(&graph, [3, 4]), 65.0),
            (subgraph(&graph, [0, 1, 2]), 100.0),
            (subgraph(&graph, [1, 2, 3]), 110.0),
            (subgraph(&graph, [2, 3, 4]), 105.0),
            (subgraph(&graph, [0, 1, 2, 3]), 150.0),
            (subgraph(&graph, [1, 2, 3, 4]), 155.0),
            (subgraph(&graph, [0, 1, 2, 3, 4]), 200.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([
                subgraph(&graph, [0, 1]),
                subgraph(&graph, [2, 3]),
                subgraph(&graph, [4]),
            ])
        );
    }

    //                /--t0--> [op1] --t3-->
    // t2 --> [op0] --+
    //                \--t1--> [op2] --t4-->
    //
    // singletons win: {op0}+{op1}+{op2}=55
    #[test]
    fn fanout2_singletons_win() {
        let graph = graph_from_edges(3, &[(0, 1), (0, 2)]);
        let costs = [
            (subgraph(&graph, [0]), 10.0),
            (subgraph(&graph, [1]), 20.0),
            (subgraph(&graph, [2]), 25.0),
            (subgraph(&graph, [0, 1]), 80.0),
            (subgraph(&graph, [0, 2]), 85.0),
            (subgraph(&graph, [0, 1, 2]), 150.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([
                subgraph(&graph, [0]),
                subgraph(&graph, [1]),
                subgraph(&graph, [2]),
            ])
        );
    }

    // t5 --> [op0] --t0--> [op1] --t1--> [op2] --t2--> [op3] --t3--> [op4] --t4--> [op5] --t6-->
    //
    // three-pair fusion: {op0,op1}+{op2,op3}+{op4,op5}=320
    #[test]
    fn chain6_three_pair_fusion() {
        let graph = graph_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let costs = [
            (subgraph(&graph, [0]), 60.0),
            (subgraph(&graph, [1]), 80.0),
            (subgraph(&graph, [2]), 50.0),
            (subgraph(&graph, [3]), 70.0),
            (subgraph(&graph, [4]), 80.0),
            (subgraph(&graph, [5]), 60.0),
            (subgraph(&graph, [0, 1]), 110.0),
            (subgraph(&graph, [1, 2]), 115.0),
            (subgraph(&graph, [2, 3]), 90.0),
            (subgraph(&graph, [3, 4]), 130.0),
            (subgraph(&graph, [4, 5]), 120.0),
            (subgraph(&graph, [0, 1, 2]), 160.0),
            (subgraph(&graph, [1, 2, 3]), 200.0),
            (subgraph(&graph, [2, 3, 4]), 210.0),
            (subgraph(&graph, [3, 4, 5]), 220.0),
            (subgraph(&graph, [0, 1, 2, 3]), 260.0),
            (subgraph(&graph, [1, 2, 3, 4]), 300.0),
            (subgraph(&graph, [2, 3, 4, 5]), 310.0),
            (subgraph(&graph, [0, 1, 2, 3, 4]), 330.0),
            (subgraph(&graph, [1, 2, 3, 4, 5]), 340.0),
            (subgraph(&graph, [0, 1, 2, 3, 4, 5]), 350.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([
                subgraph(&graph, [0, 1]),
                subgraph(&graph, [2, 3]),
                subgraph(&graph, [4, 5]),
            ])
        );
    }

    // t3 --> [op0] --t0--\
    //                    +--> [op2] --t2--> [op3] --t5-->
    // t4 --> [op1] --t1--/
    //
    // {op0,op1} disconnected → excluded from convex subgraphs
    // full fusion: {op0,op1,op2,op3}=90
    #[test]
    fn merge_full_fusion() {
        let graph = graph_from_edges(4, &[(0, 2), (1, 2), (2, 3)]);
        let convex = extract_convex_subgraphs(&graph);
        let expected_convex = HashSet::from([
            subgraph(&graph, [0]),
            subgraph(&graph, [1]),
            subgraph(&graph, [2]),
            subgraph(&graph, [3]),
            subgraph(&graph, [0, 2]),
            subgraph(&graph, [1, 2]),
            subgraph(&graph, [2, 3]),
            subgraph(&graph, [0, 1, 2]),
            subgraph(&graph, [0, 2, 3]),
            subgraph(&graph, [1, 2, 3]),
            subgraph(&graph, [0, 1, 2, 3]),
        ]);
        assert_eq!(convex, expected_convex);

        let costs = [
            (subgraph(&graph, [0]), 20.0),
            (subgraph(&graph, [1]), 15.0),
            (subgraph(&graph, [2]), 40.0),
            (subgraph(&graph, [3]), 35.0),
            (subgraph(&graph, [0, 2]), 55.0),
            (subgraph(&graph, [1, 2]), 50.0),
            (subgraph(&graph, [2, 3]), 65.0),
            (subgraph(&graph, [0, 1, 2]), 72.0),
            (subgraph(&graph, [0, 2, 3]), 82.0),
            (subgraph(&graph, [1, 2, 3]), 78.0),
            (subgraph(&graph, [0, 1, 2, 3]), 90.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(selected, HashSet::from([subgraph(&graph, [0, 1, 2, 3])]));
    }

    // t3 --> [op0] --t0--> [op1] --t1--> [op2] --t2--> [op3] --t4-->
    //
    // middle fusion: {op0}+{op1,op2}+{op3}=80
    #[test]
    fn chain4_middle_fusion() {
        let graph = graph_from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let costs = [
            (subgraph(&graph, [0]), 15.0),
            (subgraph(&graph, [1]), 40.0),
            (subgraph(&graph, [2]), 35.0),
            (subgraph(&graph, [3]), 20.0),
            (subgraph(&graph, [0, 1]), 50.0),
            (subgraph(&graph, [1, 2]), 45.0),
            (subgraph(&graph, [2, 3]), 55.0),
            (subgraph(&graph, [0, 1, 2]), 90.0),
            (subgraph(&graph, [1, 2, 3]), 88.0),
            (subgraph(&graph, [0, 1, 2, 3]), 120.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([
                subgraph(&graph, [0]),
                subgraph(&graph, [1, 2]),
                subgraph(&graph, [3]),
            ])
        );
    }

    // t3 --> [op0] --t0--> [op1] --t1--> [op2] --t2--> [op3] --t4-->
    //
    // head fusion: {op0,op1,op2}+{op3}=80
    #[test]
    fn chain4_head_fusion() {
        let graph = graph_from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let costs = [
            (subgraph(&graph, [0]), 30.0),
            (subgraph(&graph, [1]), 28.0),
            (subgraph(&graph, [2]), 26.0),
            (subgraph(&graph, [3]), 25.0),
            (subgraph(&graph, [0, 1]), 52.0),
            (subgraph(&graph, [1, 2]), 50.0),
            (subgraph(&graph, [2, 3]), 48.0),
            (subgraph(&graph, [0, 1, 2]), 55.0),
            (subgraph(&graph, [1, 2, 3]), 75.0),
            (subgraph(&graph, [0, 1, 2, 3]), 95.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0, 1, 2]), subgraph(&graph, [3])])
        );
    }

    //                /--t0--> [op1] --t3-->
    // t2 --> [op0] --+
    //                \--t1--> [op2] --t4-->
    //
    // recomputation: {op0,op1}+{op0,op2}=58, op0 recomputed
    #[test]
    fn fanout_recompute() {
        let graph = graph_from_edges(3, &[(0, 1), (0, 2)]);
        let costs = [
            (subgraph(&graph, [0]), 50.0),
            (subgraph(&graph, [1]), 40.0),
            (subgraph(&graph, [2]), 35.0),
            (subgraph(&graph, [0, 1]), 30.0),
            (subgraph(&graph, [0, 2]), 28.0),
            (subgraph(&graph, [0, 1, 2]), 80.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0, 1]), subgraph(&graph, [0, 2])])
        );
    }

    // t2 --> [op0] --t0--> [op1] --t1--> [op2] --t3-->
    //
    // recomputation: {op0,op1}+{op1,op2}=67, op1 recomputed
    #[test]
    fn chain_recompute() {
        let graph = graph_from_edges(3, &[(0, 1), (1, 2)]);
        let costs = [
            (subgraph(&graph, [0]), 40.0),
            (subgraph(&graph, [1]), 50.0),
            (subgraph(&graph, [2]), 40.0),
            (subgraph(&graph, [0, 1]), 35.0),
            (subgraph(&graph, [1, 2]), 32.0),
            (subgraph(&graph, [0, 1, 2]), 80.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0, 1]), subgraph(&graph, [1, 2])])
        );
    }

    //                /--t0--> [op1] --t4-->
    // t3 --> [op0] --+--t1--> [op2] --t5-->
    //                \--t2--> [op3] --t6-->
    //
    // op0 is multi-output (3 outputs)
    // {op1,op2}, {op1,op3}, {op2,op3}, {op1,op2,op3} disconnected → excluded
    // recomputation: {op0,op1}+{op0,op2}+{op0,op3}=75, op0 recomputed 3x
    #[test]
    fn wide_fanout_recompute() {
        let graph = graph_from_edges(4, &[(0, 1), (0, 2), (0, 3)]);
        let convex = extract_convex_subgraphs(&graph);
        let expected_convex = HashSet::from([
            subgraph(&graph, [0]),
            subgraph(&graph, [1]),
            subgraph(&graph, [2]),
            subgraph(&graph, [3]),
            subgraph(&graph, [0, 1]),
            subgraph(&graph, [0, 2]),
            subgraph(&graph, [0, 3]),
            subgraph(&graph, [0, 1, 2]),
            subgraph(&graph, [0, 1, 3]),
            subgraph(&graph, [0, 2, 3]),
            subgraph(&graph, [0, 1, 2, 3]),
        ]);
        assert_eq!(convex, expected_convex);

        let costs = [
            (subgraph(&graph, [0]), 60.0),
            (subgraph(&graph, [1]), 30.0),
            (subgraph(&graph, [2]), 30.0),
            (subgraph(&graph, [3]), 30.0),
            (subgraph(&graph, [0, 1]), 25.0),
            (subgraph(&graph, [0, 2]), 25.0),
            (subgraph(&graph, [0, 3]), 25.0),
            (subgraph(&graph, [0, 1, 2]), 80.0),
            (subgraph(&graph, [0, 1, 3]), 80.0),
            (subgraph(&graph, [0, 2, 3]), 80.0),
            (subgraph(&graph, [0, 1, 2, 3]), 100.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([
                subgraph(&graph, [0, 1]),
                subgraph(&graph, [0, 2]),
                subgraph(&graph, [0, 3]),
            ])
        );
    }

    //                /--t0--> [op1] --t2--\
    // t5 --> [op0] --+                     +--> [op3] --t4--> [op4] --t6-->
    //                \--t1--> [op2] --t3--/
    //
    // op0 is multi-output (2 outputs), op3 is multi-input (2 inputs)
    // diamond fusion + separate tail: {op0,op1,op2,op3}+{op4}=80
    #[test]
    fn diamond_tail_partial_fusion() {
        let graph = graph_from_edges(5, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]);
        let costs = [
            (subgraph(&graph, [0]), 30.0),
            (subgraph(&graph, [1]), 25.0),
            (subgraph(&graph, [2]), 25.0),
            (subgraph(&graph, [3]), 40.0),
            (subgraph(&graph, [4]), 15.0),
            (subgraph(&graph, [0, 1]), 45.0),
            (subgraph(&graph, [0, 2]), 45.0),
            (subgraph(&graph, [1, 3]), 50.0),
            (subgraph(&graph, [2, 3]), 50.0),
            (subgraph(&graph, [3, 4]), 45.0),
            (subgraph(&graph, [0, 1, 2]), 55.0),
            (subgraph(&graph, [1, 2, 3]), 60.0),
            (subgraph(&graph, [1, 3, 4]), 55.0),
            (subgraph(&graph, [2, 3, 4]), 55.0),
            (subgraph(&graph, [1, 2, 3, 4]), 70.0),
            (subgraph(&graph, [0, 1, 2, 3]), 65.0),
            (subgraph(&graph, [0, 1, 2, 3, 4]), 100.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0, 1, 2, 3]), subgraph(&graph, [4])])
        );
    }

    //                                /--t2--> [op3] --t6-->
    //                /--t0--> [op1]-+
    // t4 --> [op0] --+               \--t3--> [op4] --t7-->
    //                \--t1--> [op2] --t5-->
    //
    // op0 is multi-output (2 outputs), op1 is multi-output (2 outputs)
    // fuse right subtree: {op0,op1,op3,op4}+{op2}=85
    #[test]
    fn double_fork_subtree_fusion() {
        let graph = graph_from_edges(5, &[(0, 1), (0, 2), (1, 3), (1, 4)]);
        let convex = extract_convex_subgraphs(&graph);
        let expected_convex = HashSet::from([
            subgraph(&graph, [0]),
            subgraph(&graph, [1]),
            subgraph(&graph, [2]),
            subgraph(&graph, [3]),
            subgraph(&graph, [4]),
            subgraph(&graph, [0, 1]),
            subgraph(&graph, [0, 2]),
            subgraph(&graph, [1, 3]),
            subgraph(&graph, [1, 4]),
            subgraph(&graph, [0, 1, 2]),
            subgraph(&graph, [0, 1, 3]),
            subgraph(&graph, [0, 1, 4]),
            subgraph(&graph, [1, 3, 4]),
            subgraph(&graph, [0, 1, 2, 3]),
            subgraph(&graph, [0, 1, 2, 4]),
            subgraph(&graph, [0, 1, 3, 4]),
            subgraph(&graph, [0, 1, 2, 3, 4]),
        ]);
        assert_eq!(convex, expected_convex);

        let costs = [
            (subgraph(&graph, [0]), 40.0),
            (subgraph(&graph, [1]), 35.0),
            (subgraph(&graph, [2]), 30.0),
            (subgraph(&graph, [3]), 20.0),
            (subgraph(&graph, [4]), 20.0),
            (subgraph(&graph, [0, 1]), 55.0),
            (subgraph(&graph, [0, 2]), 50.0),
            (subgraph(&graph, [1, 3]), 40.0),
            (subgraph(&graph, [1, 4]), 40.0),
            (subgraph(&graph, [0, 1, 2]), 75.0),
            (subgraph(&graph, [0, 1, 3]), 60.0),
            (subgraph(&graph, [0, 1, 4]), 60.0),
            (subgraph(&graph, [1, 3, 4]), 45.0),
            (subgraph(&graph, [0, 1, 2, 3]), 85.0),
            (subgraph(&graph, [0, 1, 2, 4]), 85.0),
            (subgraph(&graph, [0, 1, 3, 4]), 55.0),
            (subgraph(&graph, [0, 1, 2, 3, 4]), 100.0),
        ];
        let selected = optimize_execution_plan(&graph, &costs);
        assert_eq!(
            selected,
            HashSet::from([subgraph(&graph, [0, 1, 3, 4]), subgraph(&graph, [2])])
        );
    }
}
