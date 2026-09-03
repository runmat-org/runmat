use std::collections::{HashMap, HashSet};

use crate::{
    AnalysisSequence, AnalysisValidationError, ProcedureStep, StateCompleteness, StatePersistence,
    StateSource,
};

use super::sequence_error;

pub(super) struct ValidatedGraph<'a> {
    pub(super) steps: HashMap<&'a str, &'a ProcedureStep>,
    ancestors: HashMap<&'a str, HashSet<&'a str>>,
}

impl<'a> ValidatedGraph<'a> {
    pub(super) fn is_ancestor(&self, candidate: &str, successor: &str) -> bool {
        self.ancestors
            .get(successor)
            .is_some_and(|values| values.contains(candidate))
    }
}

pub(super) fn validate(
    sequence: &AnalysisSequence,
) -> Result<ValidatedGraph<'_>, AnalysisValidationError> {
    let mut steps = HashMap::new();
    for step in &sequence.steps {
        let id = step.step_id.as_str();
        if id.trim().is_empty() {
            return Err(super::step_error(step, "step id must be non-empty"));
        }
        if steps.insert(id, step).is_some() {
            return Err(super::step_error(step, format!("duplicate step id `{id}`")));
        }
    }

    let edges = validate_dependencies(sequence, &steps)?;
    validate_acyclic(sequence, &steps, &edges)?;
    let ancestors = transitive_ancestors(&steps, &edges);
    let graph = ValidatedGraph { steps, ancestors };
    validate_branches(sequence, &graph)?;
    Ok(graph)
}

fn validate_dependencies<'a>(
    sequence: &'a AnalysisSequence,
    steps: &HashMap<&'a str, &'a ProcedureStep>,
) -> Result<HashMap<&'a str, Vec<&'a str>>, AnalysisValidationError> {
    let mut seen = HashSet::new();
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();
    for dependency in &sequence.dependencies {
        let predecessor = dependency.predecessor.as_str();
        let successor = dependency.successor.as_str();
        if !steps.contains_key(predecessor) || !steps.contains_key(successor) {
            return Err(sequence_error(
                sequence,
                format!("dependency `{predecessor}` -> `{successor}` references an unknown step"),
            ));
        }
        if predecessor == successor {
            return Err(sequence_error(
                sequence,
                format!("step `{predecessor}` cannot depend on itself"),
            ));
        }
        if !seen.insert((predecessor, successor)) {
            return Err(sequence_error(
                sequence,
                format!("duplicate dependency `{predecessor}` -> `{successor}`"),
            ));
        }
        edges.entry(predecessor).or_default().push(successor);
    }
    Ok(edges)
}

fn validate_acyclic(
    sequence: &AnalysisSequence,
    steps: &HashMap<&str, &ProcedureStep>,
    edges: &HashMap<&str, Vec<&str>>,
) -> Result<(), AnalysisValidationError> {
    fn visit<'a>(
        node: &'a str,
        edges: &HashMap<&'a str, Vec<&'a str>>,
        visiting: &mut HashSet<&'a str>,
        visited: &mut HashSet<&'a str>,
    ) -> bool {
        if visiting.contains(node) {
            return false;
        }
        if visited.contains(node) {
            return true;
        }
        visiting.insert(node);
        if edges
            .get(node)
            .into_iter()
            .flatten()
            .any(|next| !visit(next, edges, visiting, visited))
        {
            return false;
        }
        visiting.remove(node);
        visited.insert(node);
        true
    }

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    if steps
        .keys()
        .any(|id| !visit(id, edges, &mut visiting, &mut visited))
    {
        Err(sequence_error(
            sequence,
            "dependency graph contains a cycle",
        ))
    } else {
        Ok(())
    }
}

fn transitive_ancestors<'a>(
    steps: &HashMap<&'a str, &'a ProcedureStep>,
    edges: &HashMap<&'a str, Vec<&'a str>>,
) -> HashMap<&'a str, HashSet<&'a str>> {
    let mut reverse: HashMap<&str, Vec<&str>> = HashMap::new();
    for (from, successors) in edges {
        for successor in successors {
            reverse.entry(successor).or_default().push(from);
        }
    }
    let mut result = HashMap::new();
    for id in steps.keys().copied() {
        let mut found = HashSet::new();
        let mut pending = reverse.get(id).cloned().unwrap_or_default();
        while let Some(previous) = pending.pop() {
            if found.insert(previous) {
                pending.extend(reverse.get(previous).into_iter().flatten().copied());
            }
        }
        result.insert(id, found);
    }
    result
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
