use std::collections::HashSet;

use crate::{
    AnalysisSequence, AnalysisValidationError, ProcedureStep, StateCompleteness, StatePersistence,
    StateSource,
};

use super::sequence_error;

mod index;
mod owned;
mod resources;
use index::{Adjacency, StepIndex};
pub(crate) use resources::resident_bytes;
#[cfg(test)]
mod tests;
mod traversal;
pub(crate) use owned::owned_topology;

pub(crate) struct ValidatedGraph<'a> {
    pub(super) steps: StepIndex<'a>,
    predecessors: Adjacency,
}

impl<'a> ValidatedGraph<'a> {
    pub(super) fn is_ancestor(&self, candidate: &str, successor: &str) -> bool {
        let (Some(candidate), Some(successor)) =
            (self.steps.index(candidate), self.steps.index(successor))
        else {
            return false;
        };
        traversal::is_ancestor(&self.predecessors, candidate, successor)
    }
}

pub(super) fn validate(
    sequence: &AnalysisSequence,
) -> Result<ValidatedGraph<'_>, AnalysisValidationError> {
    // Check count overflow before any indexed graph allocation.
    resident_bytes(sequence)?;
    let first_empty = sequence
        .steps
        .iter()
        .position(|step| step.step_id.as_str().trim().is_empty());
    let mut entries: Vec<_> = sequence.steps.iter().enumerate().collect();
    entries.sort_unstable_by(|(a, x), (b, y)| x.step_id.cmp(&y.step_id).then(a.cmp(b)));
    if let Some((ordinal, step)) = entries
        .windows(2)
        .filter(|w| w[0].1.step_id == w[1].1.step_id)
        .map(|w| w[1])
        .min_by_key(|(i, _)| *i)
    {
        if let Some(i) = first_empty.filter(|&i| i < ordinal) {
            return Err(super::step_error(
                &sequence.steps[i],
                "step id must be non-empty",
            ));
        }
        return Err(super::step_error(
            step,
            format!("duplicate step id `{}`", step.step_id.as_str()),
        ));
    }
    if let Some(i) = first_empty {
        return Err(super::step_error(
            &sequence.steps[i],
            "step id must be non-empty",
        ));
    }
    let steps = StepIndex(entries);
    let edges = validate_dependencies(sequence, &steps)?;
    let forward = Adjacency::from_edges(sequence.steps.len(), &edges, false);
    traversal::validate_acyclic(sequence, &forward)?;
    let predecessors = Adjacency::from_edges(sequence.steps.len(), &edges, true);
    drop(forward);
    drop(edges);
    let graph = ValidatedGraph {
        steps,
        predecessors,
    };
    validate_branches(sequence, &graph)?;
    Ok(graph)
}

fn validate_dependencies(
    sequence: &AnalysisSequence,
    steps: &StepIndex<'_>,
) -> Result<Vec<(usize, usize)>, AnalysisValidationError> {
    let mut declarations: Vec<_> = sequence.dependencies.iter().enumerate().collect();
    declarations.sort_unstable_by(|(a, x), (b, y)| {
        (&x.predecessor, &x.successor)
            .cmp(&(&y.predecessor, &y.successor))
            .then(a.cmp(b))
    });
    let mut duplicates = Vec::with_capacity(declarations.len());
    for window in declarations.windows(2) {
        if window[0].1.predecessor == window[1].1.predecessor
            && window[0].1.successor == window[1].1.successor
        {
            duplicates.push(window[1].0);
        }
    }
    duplicates.sort_unstable();
    drop(declarations);
    let mut edges = Vec::with_capacity(sequence.dependencies.len());
    for (ordinal, dependency) in sequence.dependencies.iter().enumerate() {
        let predecessor = dependency.predecessor.as_str();
        let successor = dependency.successor.as_str();
        let (Some(from), Some(to)) = (steps.index(predecessor), steps.index(successor)) else {
            return Err(sequence_error(
                sequence,
                format!("dependency `{predecessor}` -> `{successor}` references an unknown step"),
            ));
        };
        if from == to {
            return Err(sequence_error(
                sequence,
                format!("step `{predecessor}` cannot depend on itself"),
            ));
        }
        if duplicates.binary_search(&ordinal).is_ok() {
            return Err(sequence_error(
                sequence,
                format!("duplicate dependency `{predecessor}` -> `{successor}`"),
            ));
        }
        edges.push((from, to));
    }
    Ok(edges)
}

fn validate_branches(
    sequence: &AnalysisSequence,
    graph: &ValidatedGraph<'_>,
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for branch in &sequence.branches {
        let id = branch.branch_id.0.as_str();
        if id.trim().is_empty() || !ids.insert(id) {
            return Err(sequence_error(
                sequence,
                format!("invalid or duplicate branch id `{id}`"),
            ));
        }
        let Some(producer) = graph.steps.get(branch.producer_step.as_str()) else {
            return Err(sequence_error(
                sequence,
                format!("branch `{id}` has unknown producer"),
            ));
        };
        if branch.state_ids.is_empty() {
            return Err(sequence_error(
                sequence,
                format!("branch `{id}` has no states"),
            ));
        }
        let mut state_ids = HashSet::new();
        for state_id in &branch.state_ids {
            if !state_ids.insert(state_id.0.as_str()) {
                return Err(sequence_error(
                    sequence,
                    format!("branch `{id}` repeats a state"),
                ));
            }
            let Some(output) = producer
                .state_outputs
                .iter()
                .find(|output| output.state_id == *state_id)
            else {
                return Err(sequence_error(
                    sequence,
                    format!(
                        "branch `{id}` references unpublished state `{}`",
                        state_id.0
                    ),
                ));
            };
            if output.completeness != StateCompleteness::Complete
                || output.persistence != StatePersistence::Canonical
            {
                return Err(sequence_error(
                    sequence,
                    format!("branch `{id}` requires complete canonical states"),
                ));
            }
        }

        let consumers: HashSet<_> = branch
            .consumer_steps
            .iter()
            .map(|value| value.as_str())
            .collect();
        if consumers.len() < 2 || consumers.len() != branch.consumer_steps.len() {
            return Err(sequence_error(
                sequence,
                format!("branch `{id}` requires at least two distinct consumers"),
            ));
        }
        for consumer in consumers {
            let Some(consumer_step) = graph.steps.get(consumer) else {
                return Err(sequence_error(
                    sequence,
                    format!("branch `{id}` has unknown consumer `{consumer}`"),
                ));
            };
            if !graph.is_ancestor(producer.step_id.as_str(), consumer) {
                return Err(sequence_error(
                    sequence,
                    format!("branch `{id}` consumer `{consumer}` is not downstream"),
                ));
            }
            let consumes_state = consumer_step.state_inputs.iter().any(|input| {
                matches!(
                    &input.source,
                    StateSource::Upstream { producer_step, state_id }
                        if producer_step == &branch.producer_step && branch.state_ids.contains(state_id)
                )
            });
            if !consumes_state {
                return Err(sequence_error(
                    sequence,
                    format!("branch `{id}` consumer `{consumer}` consumes none of its states"),
                ));
            }
        }
    }
    Ok(())
}
