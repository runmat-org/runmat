//! Isolated allocator evidence for the completed physical transient recovery preflight.

use mesh::{physical_fixture_mesh, RecoveryMesh};
use runmat_analysis_core::AnalysisModel;
use runmat_analysis_fea::{
    execution::{
        StructuralTransientBlockPlan, StructuralTransientBlockRequest,
        StructuralTransientNumericalReceipt, StructuralTransientResultAssemblyRequest,
        StructuralTransientSolverPlan, StructuralTransientSolverResult,
        StructuralTransientWorkload,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::transient::{
        prepare_transient_checkpoint_scope, recover_completed_transient_run, run_transient_block,
        CompletedStructuralTransientRunInput, StructuralTransientRunBlockInput,
    },
    planning::resource::build_structural_transient_recovery_resource_plan,
    solve::transient::StructuralTimeStepPolicy,
    ComputeBackend,
};

#[path = "transient_recovery_residency/mesh.rs"]
mod mesh;
#[path = "../support/peak_allocator.rs"]
mod peak_allocator;
use peak_allocator::{begin_measurement, end_measurement, process_peak_rss_bytes};

fn connector_observation_model() -> AnalysisModel {
    use runmat_analysis_core::*;

    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.connectors = vec![AnalysisConnector {
        connector_id: "root_to_tip_dashpot".into(),
        kind: ConnectorKind::AxialDashpot {
            primary: RegionTarget::point("node:2"),
            secondary: RegionTarget::point("node:4"),
            damping: Damping::new(75.0),
        },
    }];
    model.sequence.steps[0].observations = vec![ObservationRequest::History(ObservationSpec {
        observation_id: ObservationId("dashpot_force_history".into()),
        quantity: ObservationQuantity::ReactionForce,
        value_shape: ObservationQuantity::ReactionForce.canonical_value_shape(),
        target: ObservationTarget::Connector {
            connector_id: "root_to_tip_dashpot".into(),
        },
        source_location: FieldLocation::Connector,
        output_location: FieldLocation::Connector,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::EveryAcceptedIncrement,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::Automatic,
    })];
    model
}

fn mixed_observation_model() -> AnalysisModel {
    use runmat_analysis_core::{
        FieldLocation, ObservationId, ObservationQuantity, ObservationRequest, ObservationTarget,
        RetentionPolicy,
    };

    let mut model = connector_observation_model();
    let ObservationRequest::History(connector_force) =
        model.sequence.steps[0].observations[0].clone()
    else {
        unreachable!("connector fixture has one history")
    };
    let mut displacement = connector_force.clone();
    displacement.observation_id = ObservationId("displacement_history".into());
    displacement.quantity = ObservationQuantity::Displacement;
    displacement.value_shape = displacement.quantity.canonical_value_shape();
    displacement.target = ObservationTarget::WholeModel;
    displacement.source_location = FieldLocation::Node;
    displacement.output_location = FieldLocation::Node;

    let mut stress = connector_force.clone();
    stress.observation_id = ObservationId("stress_history".into());
    stress.quantity = ObservationQuantity::Stress;
    stress.value_shape = stress.quantity.canonical_value_shape();
    stress.target = ObservationTarget::WholeModel;
    stress.source_location = FieldLocation::Element;
    stress.output_location = FieldLocation::Element;

    let mut connector_energy = connector_force;
    connector_energy.observation_id = ObservationId("dashpot_energy_history".into());
    connector_energy.quantity = ObservationQuantity::StrainEnergy;
    connector_energy.value_shape = connector_energy.quantity.canonical_value_shape();
    connector_energy.storage.retention = RetentionPolicy::LatestSamples { count: 2 };
    model.sequence.steps[0].observations.extend([
        ObservationRequest::History(displacement),
        ObservationRequest::History(stress),
        ObservationRequest::History(connector_energy),
    ]);
    model
}

fn modal_recovery_model_and_state() -> (
    AnalysisModel,
    StructuralTransientSolverPlan,
    runmat_analysis_fea::execution::CanonicalSemanticState,
) {
    use runmat_analysis_core::{
        ProcedureStep, ProcedureStepId, ReferenceStatePolicy, SequenceDependency,
        SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod, StateUse,
        TopologyTransitionPolicy, TransitionId,
    };
    use runmat_analysis_fea::{
        execution::{
            CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
            CanonicalStateTransition, SequenceIdentity, SolverResultIdentity, StepIdentity,
        },
        ModalSolveOptions, ModalTruncationPolicy,
    };
    use runmat_execution::Digest;

    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut transient_step =
        ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
    transient_step.independent_domain = model.sequence.steps[0].independent_domain.clone();
    model.sequence.steps = vec![modal_step, transient_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("response".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-response".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("response".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let mesh = physical_fixture_mesh(&model, RecoveryMesh::Tet4);
    let modal = runmat_analysis_fea::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(1),
            solver_mesh: Some(mesh.clone()),
            ..ModalSolveOptions::default()
        },
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal.basis.identity().digest()).unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal.basis),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::bind_upstream(
        &model,
        &ProcedureStepId("response".into()),
        &state,
        StateUse::ModalBasis,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )
    .unwrap();
    (
        model,
        StructuralTransientSolverPlan::ModalReduced {
            basis_transition: Box::new(transition),
        },
        state,
    )
}

struct RecoveryMeasurement {
    peak_live_bytes: u64,
    planned_bytes: u64,
    displacement_values: usize,
}

fn measure_physical_recovery(
    case: &'static str,
    model: AnalysisModel,
    interval_count: usize,
    mesh_kind: RecoveryMesh,
    solver: StructuralTransientSolverPlan,
    modal_state: Option<&runmat_analysis_fea::execution::CanonicalSemanticState>,
) -> RecoveryMeasurement {
    let mesh = physical_fixture_mesh(&model, mesh_kind);
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        step_id: model.sequence.steps.last().unwrap().step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        solver,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
        residual_target: 1.0e-10,
    };
    let scope = prepare_transient_checkpoint_scope(&workload, &mesh, modal_state).unwrap();
    let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    let terminals = StructuralTransientBlockPlan::new(49)
        .unwrap()
        .terminal_schedule_ordinals(terminal)
        .unwrap();
    let mut predecessor = None;
    let mut segments = Vec::new();
    for terminal in terminals {
        let request =
            StructuralTransientBlockRequest::new(scope.clone(), predecessor.as_ref(), terminal)
                .unwrap();
        let segment = run_transient_block(StructuralTransientRunBlockInput {
            workload: &workload,
            solver_mesh: &mesh,
            request: &request,
            predecessor: predecessor.as_ref(),
            modal_state,
            observation_cursors: Vec::new(),
        })
        .unwrap();
        predecessor = Some(segment.manifest().terminal_checkpoint.clone());
        segments.push(segment);
    }
    let request = StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
    let result = StructuralTransientSolverResult::assemble(&request, &segments).unwrap();
    let receipt = StructuralTransientNumericalReceipt::new(&result, &request, &segments).unwrap();
    drop(segments);
    drop(predecessor);
    drop(request);
    drop(scope);

    let plan = build_structural_transient_recovery_resource_plan(
        &workload,
        &mesh,
        modal_state,
        &result,
        &receipt,
    )
    .unwrap();
    let baseline = begin_measurement();
    let recovered = recover_completed_transient_run(CompletedStructuralTransientRunInput {
        workload: &workload,
        solver_mesh: &mesh,
        modal_state,
        result,
        numerical_receipt: &receipt,
    })
    .unwrap();
    let peak = end_measurement();
    assert!(recovered.time_points_s.len() > interval_count);
    assert_eq!(
        recovered.displacement_snapshots.len(),
        recovered.time_points_s.len()
    );
    let node_count = mesh.topology.nodes.len();
    for field in &recovered.displacement_snapshots {
        assert_eq!(field.shape, [node_count, 3]);
        let values = field
            .as_host_f64()
            .expect("CPU recovery retains host fields");
        assert_eq!(values.len(), node_count * 3);
        assert!(values.iter().all(|value| value.is_finite()));
    }
    let execution_step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .unwrap();
    assert_eq!(
        recovered.run.observation_histories.len(),
        execution_step.observations.len()
    );
    for (history, request) in recovered
        .run
        .observation_histories
        .iter()
        .zip(&execution_step.observations)
    {
        let runmat_analysis_core::ObservationRequest::History(spec) = request else {
            unreachable!("recovery fixture only authors histories")
        };
        let expected = match &spec.storage.retention {
            runmat_analysis_core::RetentionPolicy::Canonical => recovered.time_points_s.len(),
            runmat_analysis_core::RetentionPolicy::LatestSamples { count } => {
                recovered.time_points_s.len().min(*count)
            }
            _ => unreachable!("recovery fixture only uses canonical or latest retention"),
        };
        assert_eq!(history.samples().len(), expected);
    }
    assert!(
        peak <= plan.request.memory_bytes,
        "measured live heap peak {peak} exceeds recovery request {} (baseline {baseline})",
        plan.request.memory_bytes
    );
    eprintln!(
        "transient recovery ({case}, {interval_count} intervals): baseline_live_heap={baseline} peak_live_heap={peak} process_peak_rss={:?} planned_memory={}",
        process_peak_rss_bytes(), plan.request.memory_bytes
    );
    RecoveryMeasurement {
        peak_live_bytes: peak,
        planned_bytes: plan.request.memory_bytes,
        displacement_values: recovered.displacement_snapshots.len() * node_count * 3,
    }
}

#[test]
fn completed_physical_recovery_heap_peak_fits_planned_memory() {
    let short = measure_physical_recovery(
        "direct",
        fixture_model(FixtureId::StructuralTransientTet4Reference),
        24,
        RecoveryMesh::Tet4,
        StructuralTransientSolverPlan::Direct,
        None,
    );
    let long = measure_physical_recovery(
        "direct",
        fixture_model(FixtureId::StructuralTransientTet4Reference),
        240,
        RecoveryMesh::Tet4,
        StructuralTransientSolverPlan::Direct,
        None,
    );
    assert!(long.peak_live_bytes > short.peak_live_bytes);
    assert!(long.planned_bytes > short.planned_bytes);
    assert!(long.displacement_values > short.displacement_values);
    let connector_observation = measure_physical_recovery(
        "connector observation",
        connector_observation_model(),
        24,
        RecoveryMesh::Tet4,
        StructuralTransientSolverPlan::Direct,
        None,
    );
    assert!(connector_observation.planned_bytes > short.planned_bytes);
    let mixed = measure_physical_recovery(
        "mixed observations",
        mixed_observation_model(),
        240,
        RecoveryMesh::Tet4,
        StructuralTransientSolverPlan::Direct,
        None,
    );
    assert!(mixed.peak_live_bytes > long.peak_live_bytes);
    assert!(mixed.planned_bytes > long.planned_bytes);
    let (modal_model, modal_solver, modal_state) = modal_recovery_model_and_state();
    measure_physical_recovery(
        "modal",
        modal_model,
        40,
        RecoveryMesh::Tet4,
        modal_solver,
        Some(&modal_state),
    );
    let tet10 = measure_physical_recovery(
        "Tet10 direct",
        fixture_model(FixtureId::StructuralTransientTet4Reference),
        24,
        RecoveryMesh::Tet10,
        StructuralTransientSolverPlan::Direct,
        None,
    );
    // Recovery consumes raw solver vectors while constructing retained fields.
    // Different element kernels need not order their transient heap peaks, but
    // all nodes/samples must survive and the admitted envelope must scale.
    assert!(tet10.displacement_values > short.displacement_values);
    assert!(tet10.planned_bytes > short.planned_bytes);
    let connected = measure_physical_recovery(
        "connected Tet4",
        fixture_model(FixtureId::StructuralTransientTet4Reference),
        24,
        RecoveryMesh::ConnectedTet4,
        StructuralTransientSolverPlan::Direct,
        None,
    );
    assert!(connected.displacement_values > short.displacement_values);
    assert!(connected.planned_bytes > short.planned_bytes);
}
