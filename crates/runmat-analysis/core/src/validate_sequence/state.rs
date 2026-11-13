use std::collections::HashSet;

use crate::{
    AnalysisModel, AnalysisProcedure, AnalysisSequence, AnalysisValidationError,
    ReferenceStatePolicy, RestartPolicy, SequenceDependencyReason, StateCompleteness,
    StatePersistence, StateSource, StateTransition, StateTransitionMethod, StateUse,
    TopologyTransitionPolicy,
};

use super::{graph::ValidatedGraph, step::family_can_consume_state, step_error, transition_error};

// Upstream contracts have already been checked by the original graph authority.
pub(super) fn validate_initial_inputs(
    step: &crate::ProcedureStep,
    model: &AnalysisModel,
) -> Result<(), AnalysisValidationError> {
    for input in &step.state_inputs {
        if matches!(input.source, StateSource::InitialConditions)
            && !model.initial_conditions.iter().any(|condition| {
                condition.applies_to_step(step)
                    && initial_condition_supplies(condition, input.required_kind)
            })
        {
            return Err(step_error(
                step,
                format!(
                    "state input `{}` has no compatible authored initial condition",
                    input.input_id,
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_input_contracts<'a>(
    step: &'a crate::ProcedureStep,
    graph: &ValidatedGraph<'a>,
) -> Result<(), AnalysisValidationError> {
    for input in &step.state_inputs {
        if let StateSource::Upstream {
            producer_step,
            state_id,
        } = &input.source
        {
            let Some(producer) = graph.steps.get(producer_step.as_str()) else {
                return Err(step_error(
                    step,
                    format!(
                        "state input `{}` references unknown producer",
                        input.input_id
                    ),
                ));
            };
            if !graph.is_ancestor(producer_step.as_str(), step.step_id.as_str()) {
                return Err(step_error(
                    step,
                    format!(
                        "state producer `{}` is not upstream",
                        producer_step.as_str()
                    ),
                ));
            }
            let Some(output) = producer
                .state_outputs
                .iter()
                .find(|value| value.state_id == *state_id)
            else {
                return Err(step_error(
                    step,
                    format!(
                        "producer `{}` does not publish state `{}`",
                        producer_step.as_str(),
                        state_id.0
                    ),
                ));
            };
            if output.kind != input.required_kind {
                return Err(step_error(
                    step,
                    format!(
                        "state input `{}` requires {:?}, but producer publishes {:?}",
                        input.input_id, input.required_kind, output.kind
                    ),
                ));
            }
            if !family_can_consume_state(step.family, input.required_kind) {
                return Err(step_error(
                    step,
                    format!(
                        "physics family {:?} cannot consume {:?} state",
                        step.family, input.required_kind
                    ),
                ));
            }
            if output.completeness != StateCompleteness::Complete
                || output.persistence == StatePersistence::CheckpointOnly
            {
                return Err(step_error(
                    step,
                    format!(
                        "state input `{}` requires a complete persisted state",
                        input.input_id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn initial_condition_supplies(condition: &crate::InitialCondition, kind: crate::StateKind) -> bool {
    matches!(
        (condition, kind),
        (
            crate::InitialCondition::Displacement(_),
            crate::StateKind::StructuralPrimary
        ) | (
            crate::InitialCondition::Velocity(_) | crate::InitialCondition::AngularVelocity(_),
            crate::StateKind::StructuralKinematics
        ) | (
            crate::InitialCondition::Temperature(_),
            crate::StateKind::Thermal
        ) | (
            crate::InitialCondition::ElectricPotential(_)
                | crate::InitialCondition::MagneticVectorPotential(_),
            crate::StateKind::ElectromagneticStatic
        ) | (
            crate::InitialCondition::AcousticPressure(_),
            crate::StateKind::AcousticComplex
        ) | (
            crate::InitialCondition::FluidPressure(_) | crate::InitialCondition::FluidVelocity(_),
            crate::StateKind::Fluid
        )
    )
}

pub(super) fn validate_transitions<'a>(
    sequence: &AnalysisSequence,
    graph: &ValidatedGraph<'a>,
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    let mut bindings = HashSet::new();
    for transition in &sequence.transitions {
        let id = transition.transition_id.0.as_str();
        if id.trim().is_empty() || !ids.insert(id) {
            return Err(transition_error(
                transition,
                "transition ids must be non-empty and unique",
            ));
        }
        let Some(producer) = graph.steps.get(transition.producer_step.as_str()) else {
            return Err(transition_error(transition, "producer step does not exist"));
        };
        let Some(consumer) = graph.steps.get(transition.consumer_step.as_str()) else {
            return Err(transition_error(transition, "consumer step does not exist"));
        };
        if producer.step_id == consumer.step_id {
            return Err(transition_error(
                transition,
                "a transition cannot target its producer",
            ));
        }
        if !graph.is_ancestor(producer.step_id.as_str(), consumer.step_id.as_str()) {
            return Err(transition_error(
                transition,
                "producer is not upstream of consumer",
            ));
        }
        let has_state_edge = sequence.dependencies.iter().any(|dependency| {
            dependency.predecessor == transition.producer_step
                && dependency.successor == transition.consumer_step
                && dependency.reason == SequenceDependencyReason::StateTransfer
        });
        if !has_state_edge {
            return Err(transition_error(
                transition,
                "transition requires a direct state-transfer dependency",
            ));
        }
        let Some(output) = producer
            .state_outputs
            .iter()
            .find(|value| value.state_id == transition.state_id)
        else {
            return Err(transition_error(
                transition,
                "producer does not publish the named state",
            ));
        };
        if output.completeness != StateCompleteness::Complete
            || output.persistence == StatePersistence::CheckpointOnly
        {
            return Err(transition_error(
                transition,
                "only a complete persisted state may cross a procedure boundary",
            ));
        }
        let matching_inputs: Vec<_> = consumer.state_inputs.iter().filter(|input| {
            input.required_kind == output.kind
                && matches!(
                    &input.source,
                    StateSource::Upstream { producer_step, state_id }
                        if producer_step == &transition.producer_step && state_id == &transition.state_id
                )
        }).collect();
        if matching_inputs.len() != 1 {
            return Err(transition_error(
                transition,
                "transition must bind exactly one matching consumer state input",
            ));
        }
        if !bindings.insert((
            transition.consumer_step.as_str(),
            transition.producer_step.as_str(),
            transition.state_id.0.as_str(),
        )) {
            return Err(transition_error(
                transition,
                "consumer state input is bound by more than one transition",
            ));
        }
        validate_transition_policy(transition, matching_inputs[0].use_as)?;
    }

    for step in graph.steps.values() {
        for input in &step.state_inputs {
            if let StateSource::Upstream {
                producer_step,
                state_id,
            } = &input.source
            {
                if !bindings.contains(&(
                    step.step_id.as_str(),
                    producer_step.as_str(),
                    state_id.0.as_str(),
                )) {
                    return Err(step_error(
                        step,
                        format!(
                            "upstream state input `{}` has no governed transition",
                            input.input_id
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_transition_policy(
    transition: &StateTransition,
    state_use: StateUse,
) -> Result<(), AnalysisValidationError> {
    if transition.method == StateTransitionMethod::Reuse
        && transition.topology != TopologyTransitionPolicy::RequireIdentical
    {
        return Err(transition_error(
            transition,
            "state reuse requires identical topology",
        ));
    }
    if transition.topology == TopologyTransitionPolicy::AdaptedMeshLineage
        && !matches!(
            transition.method,
            StateTransitionMethod::ConservativeProjection | StateTransitionMethod::Interpolation
        )
    {
        return Err(transition_error(
            transition,
            "adapted-mesh lineage requires projection or interpolation",
        ));
    }
    if (transition.method == StateTransitionMethod::InitialGuessOnly)
        != (state_use == StateUse::InitialGuess)
    {
        return Err(transition_error(
            transition,
            "initial-guess inputs and transition methods must agree",
        ));
    }
    if matches!(
        transition.method,
        StateTransitionMethod::Linearize | StateTransitionMethod::DeriveOperatorContribution
    ) && transition.reference != ReferenceStatePolicy::ExplicitLinearizationPoint
    {
        return Err(transition_error(transition, "linearization and derived-operator transitions require an explicit linearization point"));
    }
    Ok(())
}

pub(super) fn validate_restart<'a>(
    step: &'a crate::ProcedureStep,
    graph: &ValidatedGraph<'a>,
) -> Result<(), AnalysisValidationError> {
    match &step.restart {
        RestartPolicy::Fresh => Ok(()),
        RestartPolicy::ContinueFromCompletedState {
            producer_step,
            state_id,
        } => {
            let Some(producer) = graph.steps.get(producer_step.as_str()) else {
                return Err(step_error(
                    step,
                    "continuation references an unknown producer",
                ));
            };
            if !graph.is_ancestor(producer_step.as_str(), step.step_id.as_str()) {
                return Err(step_error(step, "continuation producer is not upstream"));
            }
            let complete = producer.state_outputs.iter().any(|output| {
                output.state_id == *state_id
                    && output.completeness == StateCompleteness::Complete
                    && output.persistence != StatePersistence::CheckpointOnly
            });
            if !complete {
                return Err(step_error(
                    step,
                    "continuation requires a complete persisted producer state",
                ));
            }
            let has_restart_input = step.state_inputs.iter().any(|input| {
                input.use_as == StateUse::Restart
                    && matches!(
                        &input.source,
                        StateSource::Upstream { producer_step: source_step, state_id: source_state }
                            if source_step == producer_step && source_state == state_id
                    )
            });
            if !has_restart_input {
                return Err(step_error(
                    step,
                    "continuation must declare the completed state as a restart input",
                ));
            }
            Ok(())
        }
        RestartPolicy::ResumeCheckpoint {
            checkpoint_id,
            required_state_digest,
        } => {
            if checkpoint_id.trim().is_empty() || required_state_digest.trim().is_empty() {
                Err(step_error(
                    step,
                    "checkpoint restart requires non-empty checkpoint and state-digest identities",
                ))
            } else if !matches!(
                step.procedure,
                AnalysisProcedure::TransientResponse
                    | AnalysisProcedure::Nonlinear
                    | AnalysisProcedure::Adaptive
            ) {
                Err(step_error(
                    step,
                    "only transient, nonlinear, or adaptive procedures may resume a checkpoint",
                ))
            } else {
                Ok(())
            }
        }
    }
}
