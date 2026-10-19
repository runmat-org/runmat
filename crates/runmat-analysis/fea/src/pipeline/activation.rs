use runmat_analysis_core::{AnalysisModel, ProcedureStep, StepObjectRef};
use std::borrow::Cow;

/// Build the exact constraint view consumed by one solver step.
///
/// Borrow the authored model when filtering would not change it. Only an actual removal requires
/// an owned projection for assembly APIs accepting `AnalysisModel`. No authored state is mutated.
pub(super) fn model_with_active_constraints<'a>(
    model: &'a AnalysisModel,
    step: &ProcedureStep,
) -> Cow<'a, AnalysisModel> {
    let is_active = |constraint: &runmat_analysis_core::AnalysisConstraint| {
        step.object_is_active(&StepObjectRef::Constraint(
            constraint.constraint_id().to_string(),
        ))
    };
    if step.activations.is_empty() || model.constraints.iter().all(is_active) {
        return Cow::Borrowed(model);
    }
    let mut active = model.clone();
    active.constraints.retain(is_active);
    Cow::Owned(active)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_core::{ActivationLifecycle, ProcedureStepId, StepObjectActivation};

    #[test]
    fn unchanged_constraint_views_borrow_the_complete_model() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        assert!(!model.constraints.is_empty());
        let mut step = ProcedureStep::thermal_transient("heat");
        let view = model_with_active_constraints(&model, &step);
        assert!(matches!(view, Cow::Borrowed(_)));
        assert!(std::ptr::eq(view.as_ref(), &model));
        step.activations = model
            .constraints
            .iter()
            .map(|constraint| StepObjectActivation {
                object: StepObjectRef::Constraint(constraint.constraint_id().into()),
                lifecycle: ActivationLifecycle::Activate,
            })
            .collect();
        let view = model_with_active_constraints(&model, &step);
        assert!(matches!(view, Cow::Borrowed(_)));
        assert!(std::ptr::eq(view.as_ref(), &model));
    }

    #[test]
    fn removing_constraints_owns_a_projection_without_mutating_the_author() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let original = model.clone();
        let mut step = ProcedureStep::thermal_transient("heat");
        step.activations = model
            .constraints
            .iter()
            .map(|constraint| StepObjectActivation {
                object: StepObjectRef::Constraint(constraint.constraint_id().into()),
                lifecycle: ActivationLifecycle::Deactivate {
                    from_step: ProcedureStepId("prior".into()),
                },
            })
            .collect();
        let view = model_with_active_constraints(&model, &step);
        assert!(matches!(view, Cow::Owned(_)));
        assert!(view.constraints.is_empty());
        assert_eq!(view.signals, model.signals);
        assert_eq!(model, original);
        // A closed inventory omitting constraints has the same exclusion semantics.
        step.activations = vec![StepObjectActivation {
            object: StepObjectRef::Excitation("unrelated".into()),
            lifecycle: ActivationLifecycle::Activate,
        }];
        assert!(model_with_active_constraints(&model, &step)
            .constraints
            .is_empty());
    }
}
