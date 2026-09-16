use runmat_geometry_core::UnitSystem;

use super::validate_sequence;
use crate::{
    ActivationLifecycle, AnalysisBoundary, AnalysisModel, AnalysisModelId, AnalysisProcedure,
    AnalysisValidationError, BoundaryOperator, CoordinateFrame, IndependentDomain,
    InitialCondition, InitialConditionTarget, InitialConditionValue, InitialStateConflictPolicy,
    InitialValue, PhysicsFamily, ProcedureStep, ProcedureStepId, RegionTarget, ResponseSpectrumId,
    RestartPolicy, SequenceBranch, SequenceBranchId, SequenceDependency, SequenceDependencyReason,
    SolutionSpace, StateCompleteness, StateId, StateInput, StateKind, StateOutput,
    StatePersistence, StateSource, StateUse, StepObjectActivation, StepObjectRef, Velocity,
};

fn model_with_steps(steps: Vec<ProcedureStep>) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId("model".into()),
        geometry_id: "geometry".into(),
        geometry_revision: 1,
        units: UnitSystem::Meter,
        frame: CoordinateFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        structural: None,
        thermal: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        constraints: Vec::new(),
        boundaries: Vec::new(),
        connectors: Vec::new(),
        signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        sequence: crate::AnalysisSequence::from_steps("sequence", steps),
    }
}

fn output(id: &str, kind: StateKind) -> StateOutput {
    StateOutput {
        state_id: StateId(id.into()),
        kind,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }
}

fn upstream_input(producer: &str, state: &str, kind: StateKind) -> StateInput {
    StateInput {
        input_id: format!("{state}_input"),
        required_kind: kind,
        source: StateSource::Upstream {
            producer_step: ProcedureStepId(producer.into()),
            state_id: StateId(state.into()),
        },
        use_as: StateUse::PhysicalInitialState,
        conflict: InitialStateConflictPolicy::Reject,
    }
}

fn initial_kinematics_input() -> StateInput {
    StateInput {
        input_id: "initial_kinematics".into(),
        required_kind: StateKind::StructuralKinematics,
        source: StateSource::InitialConditions,
        use_as: StateUse::PhysicalInitialState,
        conflict: InitialStateConflictPolicy::Reject,
    }
}

fn initial_velocity(applies_before_step_id: &str) -> InitialCondition {
    InitialCondition::Velocity(InitialConditionValue {
        initial_condition_id: "initial_velocity".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::UniformVector3([
            Velocity::new(1.0),
            Velocity::new(0.0),
            Velocity::new(0.0),
        ]),
        applies_before_step_id: applies_before_step_id.into(),
    })
}

#[test]
fn rejects_family_procedure_and_domain_mismatches() {
    let invalid_family = ProcedureStep::experimental(
        "thermal_modes",
        PhysicsFamily::Thermal,
        AnalysisProcedure::ModalExtraction,
        SolutionSpace::Direct,
        IndependentDomain::Eigenvalue { requested_count: 2 },
        vec![output("thermal", StateKind::Thermal)],
    );
    assert!(matches!(
        validate_sequence(&model_with_steps(vec![invalid_family])),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));

    let mut invalid_domain = ProcedureStep::structural_static("static");
    invalid_domain.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
    };
    assert!(matches!(
        validate_sequence(&model_with_steps(vec![invalid_domain])),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn modal_response_requires_typed_existing_spectrum_ids() {
    let mut step = ProcedureStep::structural_modal_response_spectrum(
        "spectrum",
        vec![ResponseSpectrumId("design".into())],
        "modes",
        "basis",
    );
    super::step::validate(&step).unwrap();
    assert_eq!(
        step.state_outputs[0].kind,
        StateKind::StructuralResponseEnvelope
    );
    let model = model_with_steps(vec![step.clone()]);
    assert!(matches!(
        super::activation::validate_objects(&model, &step),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));

    step.independent_domain = IndependentDomain::StudyParameter {
        parameter_id: "spectrum".into(),
        canonical_values: vec![1.0],
    };
    assert!(matches!(
        super::step::validate(&step),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn rejects_dependency_cycles() {
    let mut model = model_with_steps(vec![
        ProcedureStep::structural_static("base"),
        ProcedureStep::structural_static("followup"),
    ]);
    model.sequence.dependencies.push(SequenceDependency {
        predecessor: ProcedureStepId("followup".into()),
        successor: ProcedureStepId("base".into()),
        reason: SequenceDependencyReason::AuthoredOrdering,
    });
    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidSequence { .. })
    ));
}

#[test]
fn rejects_incomplete_upstream_state() {
    let mut producer = ProcedureStep::structural_static("base");
    producer.state_outputs[0].completeness = StateCompleteness::Partial;
    let mut consumer = ProcedureStep::structural_static("consumer");
    consumer.state_inputs.push(upstream_input(
        "base",
        "structural_primary",
        StateKind::StructuralPrimary,
    ));
    let mut model = model_with_steps(vec![producer, consumer]);
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;

    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn rejects_incompatible_upstream_state_kind() {
    let producer = ProcedureStep::structural_static("base");
    let mut consumer = ProcedureStep::structural_static("consumer");
    consumer.state_inputs.push(upstream_input(
        "base",
        "structural_primary",
        StateKind::Thermal,
    ));
    let mut model = model_with_steps(vec![producer, consumer]);
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;

    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn rejects_upstream_input_without_governed_transition() {
    let producer = ProcedureStep::structural_static("base");
    let mut consumer = ProcedureStep::structural_static("consumer");
    consumer.state_inputs.push(upstream_input(
        "base",
        "structural_primary",
        StateKind::StructuralPrimary,
    ));
    let mut model = model_with_steps(vec![producer, consumer]);
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;

    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn rejects_unknown_activation_objects() {
    let mut step = ProcedureStep::structural_static("static");
    step.activations.push(StepObjectActivation {
        object: StepObjectRef::Constraint("missing".into()),
        lifecycle: ActivationLifecycle::Activate,
    });
    assert!(matches!(
        validate_sequence(&model_with_steps(vec![step])),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn initial_state_input_requires_a_condition_for_the_consuming_step() {
    let first = ProcedureStep::structural_transient("first");
    let mut second = ProcedureStep::structural_transient("second");
    second.state_inputs.push(initial_kinematics_input());
    let mut model = model_with_steps(vec![first, second]);
    model.initial_conditions = vec![initial_velocity("first")];

    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn initial_state_input_requires_the_condition_to_be_active() {
    let mut step = ProcedureStep::structural_transient("transient");
    step.state_inputs.push(initial_kinematics_input());
    step.activations.push(StepObjectActivation {
        object: StepObjectRef::Boundary("wall".into()),
        lifecycle: ActivationLifecycle::Activate,
    });
    let mut model = model_with_steps(vec![step]);
    model.initial_conditions = vec![initial_velocity("transient")];
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "wall".into(),
        target: RegionTarget::boundary("wall_region"),
        operator: BoundaryOperator::AcousticRigidWall,
    });

    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn rejects_activation_lifecycle_from_nonexistent_step() {
    let object = StepObjectRef::Boundary("wall".into());
    let mut base = ProcedureStep::structural_static("base");
    base.activations.push(StepObjectActivation {
        object: object.clone(),
        lifecycle: ActivationLifecycle::Activate,
    });
    let mut consumer = ProcedureStep::structural_static("consumer");
    consumer.activations.push(StepObjectActivation {
        object,
        lifecycle: ActivationLifecycle::Continue {
            from_step: ProcedureStepId("missing".into()),
        },
    });
    let mut model = model_with_steps(vec![base, consumer]);
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "wall".into(),
        target: RegionTarget::boundary("wall_region"),
        operator: BoundaryOperator::AcousticRigidWall,
    });

    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn rejects_invalid_branches() {
    let producer = ProcedureStep::structural_static("base");
    let consumer = ProcedureStep::structural_static("consumer");
    let mut model = model_with_steps(vec![producer, consumer]);
    model.sequence.branches.push(SequenceBranch {
        branch_id: SequenceBranchId("branch".into()),
        producer_step: ProcedureStepId("base".into()),
        state_ids: vec![StateId("structural_primary".into())],
        consumer_steps: vec![ProcedureStepId("consumer".into())],
    });

    assert!(matches!(
        validate_sequence(&model),
        Err(AnalysisValidationError::InvalidSequence { .. })
    ));
}

#[test]
fn rejects_checkpoint_restart_without_stable_identity() {
    let mut step = ProcedureStep::structural_transient("transient");
    step.restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: String::new(),
        required_state_digest: String::new(),
    };
    assert!(matches!(
        validate_sequence(&model_with_steps(vec![step])),
        Err(AnalysisValidationError::InvalidProcedureStep { .. })
    ));
}

#[test]
fn exposes_deterministic_parallel_sequence_topology() {
    let mut model = model_with_steps(vec![
        ProcedureStep::structural_static("finish"),
        ProcedureStep::structural_static("right"),
        ProcedureStep::structural_static("root"),
        ProcedureStep::structural_static("left"),
    ]);
    model.sequence.dependencies = vec![
        SequenceDependency {
            predecessor: ProcedureStepId("right".into()),
            successor: ProcedureStepId("finish".into()),
            reason: SequenceDependencyReason::AuthoredOrdering,
        },
        SequenceDependency {
            predecessor: ProcedureStepId("root".into()),
            successor: ProcedureStepId("right".into()),
            reason: SequenceDependencyReason::AuthoredOrdering,
        },
        SequenceDependency {
            predecessor: ProcedureStepId("left".into()),
            successor: ProcedureStepId("finish".into()),
            reason: SequenceDependencyReason::AuthoredOrdering,
        },
        SequenceDependency {
            predecessor: ProcedureStepId("root".into()),
            successor: ProcedureStepId("left".into()),
            reason: SequenceDependencyReason::AuthoredOrdering,
        },
    ];

    let validated = validate_sequence(&model).expect("diamond sequence should validate");
    let ids = |values: &[ProcedureStepId]| {
        values
            .iter()
            .map(|id| id.as_str().to_owned())
            .collect::<Vec<_>>()
    };

    assert_eq!(
        ids(validated.topological_order()),
        vec!["root", "left", "right", "finish"]
    );
    assert_eq!(validated.parallel_levels().len(), 3);
    assert_eq!(ids(&validated.parallel_levels()[0]), vec!["root"]);
    assert_eq!(ids(&validated.parallel_levels()[1]), vec!["left", "right"]);
    assert_eq!(ids(&validated.parallel_levels()[2]), vec!["finish"]);
    assert_eq!(
        ids(&validated.direct_predecessors()[&ProcedureStepId("finish".into())]),
        vec!["left", "right"]
    );
    assert_eq!(validated.sequence(), &model.sequence);
}
