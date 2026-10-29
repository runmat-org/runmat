use super::*;
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
use runmat_meshing_core::StableDigest;

mod initial_selection;

fn compiled(step: ProcedureStep) -> CompiledAnalysisStudy {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Thermal(
        runmat_analysis_core::ThermalConstraint {
            constraint_id: "temperature".into(),
            kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                target: runmat_analysis_core::RegionTarget::authored(
                    runmat_meshing_core::fixtures::fixture_face("root"),
                ),
                value: runmat_analysis_core::PrescribedValue::Constant(Temperature::new(293.15)),
            },
        },
    )];
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps("heat-study", vec![step]);
    let properties = crate::properties::compile_property_map(&model, None).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/thermal.fea".into(),
        study_id: "thermal".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        model,
        properties,
        solver_mesh_digest: StableDigest::ZERO,
    }
}

fn policy() -> ThermalStudyPlanningPolicy {
    ThermalStudyPlanningPolicy {
        backend: ComputeBackend::Cpu,
        element_heat_source: None,
    }
}

#[test]
fn preserves_authored_transient_grid_accuracy_and_model() {
    let mut step = ProcedureStep::thermal_transient("heat");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 2.0,
        start: 2.0,
        end: 8.0,
        interval_count: 3,
    };
    step.accuracy.criteria[0].tolerance = AccuracyTolerance::Relative(3.0e-7);
    let study = compiled(step);
    let workload = plan_thermal_study(&study, policy()).unwrap();
    assert_eq!(workload.model, study.model);
    assert_eq!(
        workload.procedure,
        ThermalProcedureControl::BackwardEuler { snapshot_count: 4 }
    );
    assert_eq!(workload.residual_target, 3.0e-7);
    assert_eq!(
        workload.solve_schedule().unwrap().points_s(),
        &[2.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn plans_steady_without_inventing_a_time_domain() {
    let study = compiled(ProcedureStep::thermal_steady("equilibrium"));
    let workload = plan_thermal_study(&study, policy()).unwrap();
    assert_eq!(workload.procedure, ThermalProcedureControl::Steady);
    assert_eq!(workload.model, study.model);
}

fn mixed_study() -> CompiledAnalysisStudy {
    let mut study = compiled(ProcedureStep::thermal_steady("heat"));
    let fixture =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let mechanical = fixture.sequence.steps[0].clone();
    let mechanical_id = mechanical.step_id.0.clone();
    study.model.sequence.steps.insert(0, mechanical);
    study.model.constraints.extend(fixture.constraints);
    study.model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    study.model.signals = fixture.signals;
    study.model.excitations = fixture.excitations;
    study.model.excitation_cases = fixture.excitation_cases;
    for case in &mut study.model.excitation_cases {
        case.activation =
            runmat_analysis_core::ExcitationCaseActivation::Steps(vec![mechanical_id.clone()]);
    }
    for action in &mut study.model.excitations {
        action.excitation_mut().active_step_ids = vec![mechanical_id.clone()];
    }
    study
}

#[test]
fn selected_nonfirst_thermal_step_preserves_full_authored_identity_and_context() {
    let study = mixed_study();
    let workload = plan_thermal_step(&study, "heat", policy()).unwrap();
    assert_eq!(workload.model, study.model);
    assert_eq!(workload.step_id.0, "heat");
    let roundtrip =
        ThermalWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap()).unwrap();
    assert_eq!(roundtrip.model, study.model);
    assert_eq!(roundtrip.identity().unwrap(), workload.identity().unwrap());
    let mut changed_context = workload.clone();
    changed_context.model.sequence.steps[0].accuracy.criteria[0].tolerance =
        AccuracyTolerance::Relative(7.0e-8);
    // The surrounding authored sequence participates in the exact workload identity.
    assert_ne!(
        changed_context.identity().unwrap(),
        workload.identity().unwrap()
    );
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::UnsupportedSequence)
    ));
}

#[test]
fn selected_planner_rejects_missing_ambiguous_nonthermal_and_inbound_steps() {
    let mut study = mixed_study();
    assert!(matches!(
        plan_thermal_step(&study, "missing", policy()),
        Err(ThermalStudyPlanningError::InvalidSelection(_))
    ));
    assert!(matches!(
        plan_thermal_step(&study, &study.model.sequence.steps[0].step_id.0, policy()),
        Err(ThermalStudyPlanningError::UnsupportedSequence)
    ));
    study
        .model
        .sequence
        .steps
        .push(study.model.sequence.steps[1].clone());
    assert!(matches!(
        plan_thermal_step(&study, "heat", policy()),
        Err(ThermalStudyPlanningError::InvalidSelection(_))
    ));
    study.model.sequence.steps.pop();
    study
        .model
        .sequence
        .dependencies
        .push(runmat_analysis_core::SequenceDependency {
            predecessor: study.model.sequence.steps[0].step_id.clone(),
            successor: study.model.sequence.steps[1].step_id.clone(),
            reason: runmat_analysis_core::SequenceDependencyReason::AuthoredOrdering,
        });
    assert!(matches!(
        plan_thermal_step(&study, "heat", policy()),
        Err(ThermalStudyPlanningError::UnsupportedSequence)
    ));
    study.model.sequence.dependencies.clear();
    study.model.sequence.steps[1].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "checkpoint".into(),
        required_state_digest: "digest".into(),
    };
    assert!(matches!(
        plan_thermal_step(&study, "heat", policy()),
        Err(ThermalStudyPlanningError::UnsupportedSequence)
    ));
}

#[test]
fn selected_planner_rejects_realized_nonthermal_loads_on_selected_step() {
    let mut study = mixed_study();
    for case in &mut study.model.excitation_cases {
        case.activation = runmat_analysis_core::ExcitationCaseActivation::Always;
    }
    for action in &mut study.model.excitations {
        action.excitation_mut().active_step_ids.clear();
    }
    assert!(matches!(
        plan_thermal_step(&study, "heat", policy()),
        Err(ThermalStudyPlanningError::Source(_))
    ));
}

#[test]
fn rejects_multiple_steps_and_upstream_state() {
    let mut study = compiled(ProcedureStep::thermal_steady("first"));
    study
        .model
        .sequence
        .steps
        .push(ProcedureStep::thermal_steady("second"));
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::UnsupportedSequence)
    ));
    study.model.sequence.steps.pop();
    study.model.sequence.steps[0].state_inputs =
        ProcedureStep::structural_modal_harmonic("response", "modes", "basis").state_inputs;
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::UnsupportedSequence)
    ));
}

#[test]
fn rejects_overflow_and_sample_ceiling_instead_of_coarsening() {
    let mut study = compiled(ProcedureStep::thermal_transient("heat"));
    let IndependentDomain::TimeSeconds { interval_count, .. } =
        &mut study.model.sequence.steps[0].independent_domain
    else {
        panic!("time domain")
    };
    *interval_count = usize::MAX;
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::SampleCountOverflow)
    ));
    study.model.sequence.steps[0] = ProcedureStep::thermal_transient("heat");
    study.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 3;
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::Schedule(_))
    ));
}

#[test]
fn rejects_unimplemented_observations_instead_of_dropping_them() {
    let mut study = compiled(ProcedureStep::thermal_steady("heat"));
    study.model.sequence.steps[0]
        .observations
        .push(ObservationRequest::Diagnostic(
            runmat_analysis_core::DiagnosticObservation {
                observation_id: runmat_analysis_core::ObservationId("balance".into()),
                diagnostic: runmat_analysis_core::DiagnosticKind::Balance,
                sampling: SamplingPolicy::Final,
                retention: runmat_analysis_core::RetentionPolicy::Canonical,
            },
        ));
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::Observation(_))
    ));
}

#[test]
fn rejects_nonthermal_actions_instead_of_ignoring_them() {
    let mut study = compiled(ProcedureStep::thermal_steady("heat"));
    let fixture =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    study.model.signals = fixture.signals;
    study.model.excitations = fixture.excitations;
    study.model.excitation_cases = fixture.excitation_cases;
    for case in &mut study.model.excitation_cases {
        case.activation = runmat_analysis_core::ExcitationCaseActivation::Always;
    }
    for action in &mut study.model.excitations {
        action.excitation_mut().active_step_ids.clear();
    }
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::Source(_))
    ));
}

#[test]
fn requires_authored_residual_target() {
    let mut study = compiled(ProcedureStep::thermal_steady("heat"));
    study.model.sequence.steps[0].accuracy.criteria.clear();
    assert!(matches!(
        plan_thermal_study(&study, policy()),
        Err(ThermalStudyPlanningError::MissingRelativeAlgebraicTarget)
    ));
}

#[test]
fn shared_heat_source_is_preserved_and_unmaterialized_signal_is_rejected() {
    use runmat_analysis_core::*;
    let mut study = compiled(ProcedureStep::thermal_transient("heat"));
    let fixture =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let mut action = fixture.excitations[0].excitation().clone();
    action.target = ExcitationTarget::WholeModel;
    action.direction = None;
    action.distribution = ExcitationDistribution::IntensityPerVolume;
    action.active_step_ids.clear();
    action.projection.preserve_moment = false;
    let PhysicalSignal::Force(old) = &fixture.signals[0] else {
        panic!("fixture signal")
    };
    study.model.signals = vec![PhysicalSignal::HeatGenerationDensity(Signal {
        signal_id: action.signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(
            HeatGenerationDensity::new(12.0),
        )),
        precision: old.precision,
        evidence: old.evidence.clone(),
        provenance: old.provenance.clone(),
    })];
    study.model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("heat-case".into()),
        action_ids: vec![action.excitation_id.clone()],
        activation: ExcitationCaseActivation::Always,
    }];
    study.model.excitations = vec![ExcitationAction::HeatGeneration(action)];
    study.model.sequence.steps[0].activations = vec![
        StepObjectActivation {
            object: StepObjectRef::Constraint("temperature".into()),
            lifecycle: ActivationLifecycle::Activate,
        },
        StepObjectActivation {
            object: StepObjectRef::Excitation(
                study.model.excitations[0]
                    .excitation()
                    .excitation_id
                    .0
                    .clone(),
            ),
            lifecycle: ActivationLifecycle::Activate,
        },
    ];
    assert_eq!(
        plan_thermal_study(&study, policy()).unwrap().model,
        study.model
    );
    let PhysicalSignal::HeatGenerationDensity(signal) = &mut study.model.signals[0] else {
        panic!("heat signal")
    };
    signal.representation = SignalRepresentation::Artifact(SignalArtifactRef {
        artifact_id: "external-history".into(),
        content_digest: "sha256:unresolved".into(),
        schema: "signal/v1".into(),
        domain: SignalArtifactDomain::TimeSeconds {
            start: 0.0,
            end: 1.0,
        },
        topology_digest: None,
        sample_count: 11,
    });
    assert!(plan_thermal_study(&study, policy()).is_err());
}
