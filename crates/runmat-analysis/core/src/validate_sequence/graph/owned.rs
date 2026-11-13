//! Materialize planning topology only after original graph validation.
use super::ValidatedGraph;
use crate::{AnalysisSequence, ProcedureStepId, ValidatedAnalysisSequence};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn owned_topology(
    sequence: &AnalysisSequence,
    graph: &ValidatedGraph<'_>,
) -> ValidatedAnalysisSequence {
    let mut direct_predecessors = BTreeMap::new();
    for step_id in graph.steps.keys() {
        direct_predecessors.insert(ProcedureStepId(step_id.to_owned()), Vec::new());
    }
    for dependency in &sequence.dependencies {
        direct_predecessors
            .get_mut(&dependency.successor)
            .expect("validated dependency successor must exist")
            .push(dependency.predecessor.clone());
    }
    for predecessors in direct_predecessors.values_mut() {
        predecessors.sort();
    }

    let mut successors: BTreeMap<ProcedureStepId, Vec<ProcedureStepId>> = graph
        .steps
        .keys()
        .map(|id| (ProcedureStepId(id.to_owned()), Vec::new()))
        .collect();
    for dependency in &sequence.dependencies {
        successors
            .get_mut(&dependency.predecessor)
            .expect("validated dependency predecessor must exist")
            .push(dependency.successor.clone());
    }
    for values in successors.values_mut() {
        values.sort();
    }

    let mut remaining_predecessors: BTreeMap<_, _> = direct_predecessors
        .iter()
        .map(|(id, predecessors)| (id.clone(), predecessors.len()))
        .collect();
    let mut ready: BTreeSet<_> = remaining_predecessors
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut topological_order = Vec::with_capacity(sequence.steps.len());
    let mut parallel_levels = Vec::new();
    while !ready.is_empty() {
        let level: Vec<_> = ready.iter().cloned().collect();
        ready.clear();
        for step_id in &level {
            topological_order.push(step_id.clone());
            for successor in successors
                .get(step_id)
                .expect("validated step must have a successor inventory")
            {
                let count = remaining_predecessors
                    .get_mut(successor)
                    .expect("validated successor must have a predecessor count");
                *count -= 1;
                if *count == 0 {
                    ready.insert(successor.clone());
                }
            }
        }
        parallel_levels.push(level);
    }
    debug_assert_eq!(topological_order.len(), sequence.steps.len());

    ValidatedAnalysisSequence::new(
        sequence.clone(),
        topological_order,
        parallel_levels,
        direct_predecessors,
    )
}
