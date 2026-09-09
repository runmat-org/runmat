use runmat_analysis_core::{AnalysisModel, ProcedureStep, StepObjectRef};

/// Build the exact constraint view consumed by one solver step.
///
/// Assembly APIs currently accept an `AnalysisModel`; this one-time projection keeps their input
/// consistent with the sequence lifecycle contract until those APIs accept a resolved step view
/// directly. No authored model state is mutated.
pub(super) fn model_with_active_constraints(
    model: &AnalysisModel,
    step: &ProcedureStep,
) -> AnalysisModel {
    let mut active = model.clone();
    active.constraints.retain(|constraint| {
        step.object_is_active(&StepObjectRef::Constraint(
            constraint.constraint_id().to_string(),
        ))
    });
    active
}
