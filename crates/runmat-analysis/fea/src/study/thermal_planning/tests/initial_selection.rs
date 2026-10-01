use super::*;
use runmat_analysis_core::{
    ActivationLifecycle, CoordinateFrame, InitialCondition, InitialConditionTarget,
    InitialConditionValue, InitialStateConflictPolicy, InitialValue, ProcedureStepId, StateId,
    StateInput, StateSource, StateUse, StepObjectActivation, StepObjectRef,
};

fn initial_study() -> CompiledAnalysisStudy {
    let mut step = ProcedureStep::thermal_transient("heat");
    step.state_inputs.push(StateInput {
        input_id: "initial_temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::InitialConditions,
        use_as: StateUse::PhysicalInitialState,
        conflict: InitialStateConflictPolicy::Reject,
    });
    step.activations.push(StepObjectActivation {
        object: StepObjectRef::InitialCondition("chosen".into()),
        lifecycle: ActivationLifecycle::Activate,
    });
    let mut study = compiled(step);
    study.model.initial_conditions = ["chosen", "unselected"]
        .into_iter()
        .map(|id| {
            InitialCondition::Temperature(InitialConditionValue {
                initial_condition_id: id.into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::Uniform(Temperature::new(293.15)),
                applies_before_step_id: "heat".into(),
            })
        })
        .collect();
    study
}

#[test]
fn authored_physical_initial_selection_preserves_kernel_selection_authority() {
    let study = initial_study();
    let workload = plan_thermal_study(&study, policy()).unwrap();
    let step = &workload.model.sequence.steps[0];
    assert_eq!(workload.model, study.model);
    assert!(workload.model.initial_conditions[0].applies_to_step(step));
    assert!(!workload.model.initial_conditions[1].applies_to_step(step));
    assert_eq!(
        crate::physics::thermal::initial_state::selected_condition_count(&workload.model, step),
        1
    );
}

#[test]
fn fresh_constraint_activations_are_not_initial_condition_selections() {
    for step in [
        ProcedureStep::thermal_steady("heat"),
        ProcedureStep::thermal_transient("heat"),
    ] {
        let mut study = compiled(step);
        study.model.sequence.steps[0]
            .activations
            .push(StepObjectActivation {
                object: StepObjectRef::Constraint("temperature".into()),
                lifecycle: ActivationLifecycle::Activate,
            });
        let workload = plan_thermal_study(&study, policy()).unwrap();
        assert_eq!(workload.model, study.model);
        assert!(workload.model.sequence.steps[0].state_inputs.is_empty());
        assert!(workload.model.sequence.steps[0]
            .object_is_active(&StepObjectRef::Constraint("temperature".into())));
    }
}

#[test]
fn mixed_constraint_and_initial_activations_preserve_closed_selection() {
    let mut study = initial_study();
    study.model.sequence.steps[0].activations.insert(
        0,
        StepObjectActivation {
            object: StepObjectRef::Constraint("temperature".into()),
            lifecycle: ActivationLifecycle::Activate,
        },
    );
    let workload = plan_thermal_study(&study, policy()).unwrap();
    let step = &workload.model.sequence.steps[0];
    assert_eq!(workload.model, study.model);
    assert!(step.object_is_active(&StepObjectRef::Constraint("temperature".into())));
    assert!(workload.model.initial_conditions[0].applies_to_step(step));
    assert!(!workload.model.initial_conditions[1].applies_to_step(step));
    assert_eq!(
        crate::physics::thermal::initial_state::selected_condition_count(&workload.model, step),
        1
    );
    // Unrelated activations must not hide malformed initial selections.
    study.model.sequence.steps[0].activations[1].object =
        StepObjectRef::InitialCondition("missing".into());
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::UnsupportedSequence)
    ));
}

#[test]
fn unsupported_initial_input_sources_kinds_uses_and_conflicts_stay_closed() {
    let baseline = initial_study();
    let mut variants = Vec::new();
    let mut wrong = baseline.clone();
    wrong.model.sequence.steps[0].state_inputs.clear();
    variants.push(wrong);
    let mut wrong = baseline.clone();
    wrong.model.sequence.steps[0].state_inputs[0].source = StateSource::Upstream {
        producer_step: ProcedureStepId("other".into()),
        state_id: StateId("state".into()),
    };
    variants.push(wrong);
    for kind in [StateKind::StructuralPrimary, StateKind::AcousticComplex] {
        let mut wrong = baseline.clone();
        wrong.model.sequence.steps[0].state_inputs[0].required_kind = kind;
        variants.push(wrong);
    }
    for use_as in [
        StateUse::InitialGuess,
        StateUse::Restart,
        StateUse::LinearizationPoint,
    ] {
        let mut wrong = baseline.clone();
        wrong.model.sequence.steps[0].state_inputs[0].use_as = use_as;
        variants.push(wrong);
    }
    let mut wrong = baseline.clone();
    wrong.model.sequence.steps[0].state_inputs[0].conflict =
        InitialStateConflictPolicy::ReplaceInitialCondition;
    variants.push(wrong);
    for study in variants {
        assert!(matches!(
            plan_thermal_study(&study, policy()),
            Err(ThermalStudyPlanningError::UnsupportedSequence)
        ));
    }
}

#[test]
fn missing_wrong_step_duplicate_and_noninitial_lifecycle_selections_stay_closed() {
    let baseline = initial_study();
    let mut variants = Vec::new();
    let mut wrong = baseline.clone();
    wrong.model.sequence.steps[0].activations[0].object =
        StepObjectRef::InitialCondition("missing".into());
    variants.push(wrong);
    let mut wrong = baseline.clone();
    let InitialCondition::Temperature(condition) = &mut wrong.model.initial_conditions[0] else {
        unreachable!()
    };
    condition.applies_before_step_id = "other".into();
    variants.push(wrong);
    let mut wrong = baseline.clone();
    let duplicate = wrong.model.sequence.steps[0].activations[0].clone();
    wrong.model.sequence.steps[0].activations.push(duplicate);
    variants.push(wrong);
    for lifecycle in [
        ActivationLifecycle::Continue {
            from_step: ProcedureStepId("other".into()),
        },
        ActivationLifecycle::Deactivate {
            from_step: ProcedureStepId("other".into()),
        },
        ActivationLifecycle::Modify {
            prior_step: ProcedureStepId("other".into()),
            prior_object: StepObjectRef::InitialCondition("prior".into()),
        },
    ] {
        let mut wrong = baseline.clone();
        wrong.model.sequence.steps[0].activations[0].lifecycle = lifecycle;
        variants.push(wrong);
    }
    let mut wrong = baseline;
    wrong.model.sequence.steps[0].activations[0].object =
        StepObjectRef::MaterialParameter("parameter".into());
    variants.push(wrong);
    for study in variants {
        assert!(matches!(
            plan_thermal_study(&study, policy()),
            Err(ThermalStudyPlanningError::UnsupportedSequence)
        ));
    }
}
