//! Bounded authored initial selection; numerical realization stays with the thermal kernel.
use runmat_analysis_core::{
    ActivationLifecycle, AnalysisModel, InitialCondition, InitialStateConflictPolicy,
    ProcedureStep, StateKind, StateSource, StateUse, StepObjectRef,
};

/// Pre-mesh family gate. Model-dependent IDs and values remain core/planning admission work.
pub fn thermal_step_initial_selection_supported(step: &ProcedureStep) -> bool {
    step.state_inputs.len() <= 1
        && step.state_inputs.iter().all(|input| {
            input.source == StateSource::InitialConditions
                && input.required_kind == StateKind::Thermal
                && input.use_as == StateUse::PhysicalInitialState
                && input.conflict == InitialStateConflictPolicy::Reject
        })
        // Loads, constraints and boundaries are not initial-state selections.
        // Their identities/lifecycles remain shared sequence admission work.
        && step.activations.iter().filter(|activation| {
            matches!(&activation.object, StepObjectRef::InitialCondition(_))
        }).all(|activation| {
            !step.state_inputs.is_empty()
                && activation.lifecycle == ActivationLifecycle::Activate
        })
}

pub(super) fn supported(model: &AnalysisModel, step: &ProcedureStep) -> bool {
    thermal_step_initial_selection_supported(step)
        && step
            .activations
            .iter()
            .enumerate()
            .filter(|(_, activation)| {
                matches!(&activation.object, StepObjectRef::InitialCondition(_))
            })
            .all(|(index, activation)| {
                !step.activations[..index]
                    .iter()
                    .any(|other| other.object == activation.object)
                    && model.initial_conditions.iter().any(|condition| {
                        matches!(condition, InitialCondition::Temperature(_))
                            && condition.applies_to_step(step)
                            && activation.object
                                == StepObjectRef::InitialCondition(
                                    condition.initial_condition_id().to_owned(),
                                )
                    })
            })
        && (step.state_inputs.is_empty()
            || model.initial_conditions.iter().any(|condition| {
                matches!(condition, InitialCondition::Temperature(_))
                    && condition.applies_to_step(step)
            }))
}
