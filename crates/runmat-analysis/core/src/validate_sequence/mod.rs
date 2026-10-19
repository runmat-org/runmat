mod activation;
pub(crate) mod graph;
mod state;
mod step;

#[cfg(test)]
mod tests;

use crate::{
    AnalysisModel, AnalysisSequence, AnalysisValidationError, PhysicsFamily, ProcedureStep,
    StateKind, StateTransition, ValidatedAnalysisSequence,
};

pub(crate) fn family_can_consume_state(family: PhysicsFamily, kind: StateKind) -> bool {
    step::family_can_consume_state(family, kind)
}

pub(crate) fn validate_sequence(
    model: &AnalysisModel,
) -> Result<graph::ValidatedGraph<'_>, AnalysisValidationError> {
    let sequence = &model.sequence;
    let validated_graph = validate_contract_graph(sequence)?;
    for procedure_step in &sequence.steps {
        activation::validate_objects(model, procedure_step)?;
        crate::validate_outputs::validate_outputs(model, procedure_step)?;
        state::validate_initial_inputs(procedure_step, model)?;
    }
    Ok(validated_graph)
}

pub(crate) fn validate_sequence_contract(
    sequence: &AnalysisSequence,
) -> Result<ValidatedAnalysisSequence, AnalysisValidationError> {
    let graph = validate_contract_graph(sequence)?;
    Ok(graph::owned_topology(sequence, &graph))
}

fn validate_contract_graph(
    sequence: &AnalysisSequence,
) -> Result<graph::ValidatedGraph<'_>, AnalysisValidationError> {
    if sequence.sequence_id.trim().is_empty() {
        return Err(sequence_error(sequence, "sequence id must be non-empty"));
    }
    if sequence.steps.is_empty() {
        return Err(sequence_error(
            sequence,
            "sequence must contain at least one procedure step",
        ));
    }

    for procedure_step in &sequence.steps {
        step::validate(procedure_step)?;
        crate::validate_accuracy::validate_accuracy(procedure_step)?;
    }

    let validated_graph = graph::validate(sequence)?;
    for procedure_step in &sequence.steps {
        activation::validate_lifecycles(procedure_step, &validated_graph)?;
        state::validate_input_contracts(procedure_step, &validated_graph)?;
        state::validate_restart(procedure_step, &validated_graph)?;
    }
    state::validate_transitions(sequence, &validated_graph)?;
    Ok(validated_graph)
}

fn sequence_error(
    sequence: &AnalysisSequence,
    detail: impl Into<String>,
) -> AnalysisValidationError {
    AnalysisValidationError::InvalidSequence {
        sequence_id: sequence.sequence_id.clone(),
        detail: detail.into(),
    }
}

fn step_error(step: &ProcedureStep, detail: impl Into<String>) -> AnalysisValidationError {
    AnalysisValidationError::InvalidProcedureStep {
        step_id: step.step_id.0.clone(),
        detail: detail.into(),
    }
}

fn transition_error(
    transition: &StateTransition,
    detail: impl Into<String>,
) -> AnalysisValidationError {
    AnalysisValidationError::InvalidStateTransition {
        transition_id: transition.transition_id.0.clone(),
        detail: detail.into(),
    }
}
