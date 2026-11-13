use runmat_analysis_core::{
    AnalysisConstraint, ExcitationTarget, IndependentDomain, PrescribedStructuralDofs,
    ProcedureStep, ProcedureStepId, ReferenceStatePolicy, RegionTarget, SequenceDependency,
    SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    StructuralConstraint, StructuralConstraintKind, TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::{
    bind_upstream_state_transition, complete_structural_transient_sequence_preparation,
    execute_serial_analysis, import_structural_transient_block, import_structural_transient_result,
    import_structural_transient_scope, prepare_analysis_objects,
    prepare_structural_transient_block_request_objects, prepare_structural_transient_block_task,
    prepare_structural_transient_checkpoint_objects, prepare_structural_transient_result_task,
    prepare_structural_transient_scope_task, prepare_structural_transient_sequence,
    prepare_structural_transient_workload_objects, AnalysisHostResponse, AnalysisKernel,
    AnalysisKernelInvocation, PreparedStructuralTransientBlockTask,
    PreparedStructuralTransientResultTask, StructuralModalTransientStudyCoordinator,
    StructuralModalTransientStudyPreparation, StructuralTransientBlockPreparation,
    StructuralTransientResultPreparation, StructuralTransientScopeTaskPreparation,
    StructuralTransientSequenceCoordinator, StructuralTransientSequencePreparation,
    StructuralTransientSequencePreparationInput,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisTaskIdentity, AnalysisWorkKind,
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        SequenceIdentity, SolverResultIdentity, StepIdentity, StructuralModalTransientTemplate,
        StructuralModalWorkload, StructuralModalWorkloadInput, StructuralStepWorkload,
        StructuralStudyWorkloads, StructuralTransientBlockPlan, StructuralTransientBlockRequest,
        StructuralTransientSolverPlan, StructuralTransientWorkload,
    },
    fixtures::{fixture_model, FixtureId},
    planning::resource::{build_resource_plan, ByteCount, FeaResourceDemand, FeaResourcePhase},
    solve::transient::StructuralTimeStepPolicy,
    ComputeBackend, ModalSolveOptions, ModalTruncationPolicy,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
};
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder};

use super::NativeAnalysisKernel;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "study:transient-block".into(),
        encryption_context: digest("transient-block-encryption"),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("transient-block-graph"),
        digest("transient-block-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn transient_fixture() -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    *model.material_assignments[0]
        .target
        .selected_region_id_mut()
        .expect("fixture uses selected material region") =
        runmat_meshing_core::fixtures::fixture_region("region");
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "fixed-face".into(),
        frame: runmat_analysis_core::CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
            translations: [
                runmat_analysis_core::StructuralTranslationDof::Ux,
                runmat_analysis_core::StructuralTranslationDof::Uy,
                runmat_analysis_core::StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .map(|dof| runmat_analysis_core::TranslationDofValue {
                dof,
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Displacement::new(0.0),
                ),
            })
            .collect(),
            rotations: Vec::new(),
        }),
    })];
    model.excitations[0].excitation_mut().target = ExcitationTarget::Region(
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:3")),
    );
    (
        model,
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
    )
}

fn modal_transient_fixture() -> (
    StructuralTransientWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalSemanticState,
) {
    let (model, mesh) = authored_modal_transient_fixture();
    prepared_modal_transient_fixture(model, mesh)
}

fn modal_support_motion_fixture() -> (
    StructuralTransientWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalSemanticState,
) {
    let (model, mesh) = authored_modal_support_motion_fixture();
    prepared_modal_transient_fixture(model, mesh)
}

fn prepared_modal_transient_fixture(
    model: runmat_analysis_core::AnalysisModel,
    mesh: runmat_meshing_core::SolverMeshArtifact,
) -> (
    StructuralTransientWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalSemanticState,
) {
    let modal = runmat_analysis_fea::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(3),
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
    let basis_transition = bind_upstream_state_transition(
        &model,
        &ProcedureStepId("response".into()),
        &state,
        runmat_analysis_core::StateUse::ModalBasis,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )
    .unwrap();
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("response".into()),
        backend: ComputeBackend::Cpu,
        solver: StructuralTransientSolverPlan::ModalReduced {
            basis_transition: Box::new(basis_transition),
        },
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-8,
    };
    (workload, mesh, state)
}

fn authored_modal_transient_fixture() -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let (mut model, mesh) = transient_fixture();
    author_modal_transient_sequence(&mut model, "modal-basis-to-transient");
    (model, mesh)
}

fn author_modal_transient_sequence(
    model: &mut runmat_analysis_core::AnalysisModel,
    transition_id: &str,
) {
    let mut modal_step = ProcedureStep::structural_modal("modes");
    modal_step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let mut transient_step =
        ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
    transient_step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 10,
    };
    model.sequence.steps = vec![modal_step, transient_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("response".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId(transition_id.into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("response".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
}

fn authored_modal_support_motion_fixture() -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    use runmat_analysis_core::*;

    let (mut model, mesh) = transient_fixture();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    let acceleration_id = SignalId("root-x-acceleration".into());
    let acceleration = |value: f64| {
        SignalOrdinate::Real(SignalValue::Vector3([
            Acceleration::new(value),
            Acceleration::new(0.0),
            Acceleration::new(0.0),
        ]))
    };
    model.signals.push(PhysicalSignal::Acceleration(Signal {
        signal_id: acceleration_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(0.0, acceleration(1.0)),
                SignalKnot::continuous(0.5, acceleration(-1.0)),
                SignalKnot::continuous(1.0, acceleration(1.0)),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 3,
        },
        provenance: SignalProvenance::Authored,
    }));
    model.constraints = vec![
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "moving-root-x".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                translations: vec![TranslationDofMotion {
                    dof: StructuralTranslationDof::Ux,
                    history: StructuralKinematicHistory::AccelerationDriven {
                        acceleration: AccelerationSignalId(acceleration_id),
                        initial_displacement: Displacement::new(0.02),
                        initial_velocity: Velocity::new(-0.03),
                        reconstruction:
                            StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
                    },
                }],
            }),
        }),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "fixed-root-yz".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                translations: [StructuralTranslationDof::Uy, StructuralTranslationDof::Uz]
                    .into_iter()
                    .map(|dof| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    })
                    .collect(),
                rotations: Vec::new(),
            }),
        }),
    ];
    author_modal_transient_sequence(&mut model, "modal-basis-to-moving-response");
    (model, mesh)
}

fn execute_durable_block(
    store: &mut runmat_execution_artifact::cache::FilesystemObjectStore,
    prepared: &PreparedStructuralTransientBlockTask,
) -> runmat_execution_artifact::ProgramExecutionResponse {
    let limits = ObjectInventoryLimits::default();
    prepared.program.write_to(store).unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response()
}

fn execute_durable_result(
    store: &mut runmat_execution_artifact::cache::FilesystemObjectStore,
    prepared: &PreparedStructuralTransientResultTask,
) -> runmat_execution_artifact::ProgramExecutionResponse {
    let limits = ObjectInventoryLimits::default();
    prepared.program.write_to(store).unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response()
}

#[test]
fn native_scope_task_rederives_its_plan_and_returns_the_exact_restart_scope() {
    let (model, mesh) = transient_fixture();
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        solver: StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    assert_scope_task_round_trip(workload, mesh, None);
}

#[test]
fn native_modal_scope_task_binds_the_published_basis_state() {
    let (workload, mesh, modal_state) = modal_transient_fixture();
    assert_scope_task_round_trip(workload, mesh, Some(modal_state));
}

fn assert_scope_task_round_trip(
    workload: StructuralTransientWorkload,
    mesh: runmat_meshing_core::SolverMeshArtifact,
    modal_state: Option<CanonicalSemanticState>,
) {
    let limits = ObjectInventoryLimits::default();
    let prepared =
        prepare_structural_transient_scope_task(StructuralTransientScopeTaskPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            workload: workload.clone(),
            mesh: mesh.clone(),
            modal_state: modal_state.clone(),
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    assert_eq!(
        prepared.host.workload.kind,
        AnalysisWorkKind::PrepareStructuralTransientCheckpointScope
    );
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path(),
        limits.max_object_bytes,
    )
    .unwrap();
    prepared.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response();
    let imported = import_structural_transient_scope(&store, response, &prepared, limits).unwrap();
    let direct = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload,
        &mesh,
        modal_state.as_ref(),
    )
    .unwrap();
    assert_eq!(imported.scope, direct);
    assert!(prepared.resource_plan.request.max_artifact_bytes > 0);
}

#[test]
fn native_kernel_executes_a_two_block_physical_transient_chain() {
    let (model, mesh) = transient_fixture();
    let step = &model.sequence.steps[0];
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        model: model.clone(),
        step_id: step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        solver: runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    chain::assert_two_block_chain(workload, mesh);
}

#[test]
fn transient_sequence_coordinator_admits_each_exact_transition_before_assembly() {
    let (model, mesh) = transient_fixture();
    let step_id = model.sequence.steps[0].step_id.clone();
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        solver: runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload, &mesh, None,
    )
    .unwrap();
    let limits = ObjectInventoryLimits::default();
    let workload_objects = prepare_structural_transient_workload_objects(
        &workload,
        scope.source_workload,
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh.canonical_encode().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let preparation = StructuralTransientSequencePreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload: workload_objects,
        mesh: mesh_objects,
        modal_state: None,
        scope,
        terminal_schedule_ordinals: vec![2, 4],
        initial_dependencies: Vec::new(),
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    };
    let mut coordinator = StructuralTransientSequenceCoordinator::new(preparation.clone()).unwrap();
    assert!(coordinator.prepare_assembly().is_err());

    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let first = coordinator.prepare_next_block().unwrap();
    assert!(coordinator.prepare_next_block().is_err());
    let first_response = execute_durable_block(&mut store, &first);
    coordinator
        .complete_block(&store, first_response, &first)
        .unwrap();
    assert_eq!(coordinator.completed_blocks().len(), 1);
    let checkpoint = coordinator.checkpoint().unwrap();
    let checkpoint =
        runmat_analysis_execution::StructuralTransientExecutionCheckpoint::from_canonical_bytes(
            &checkpoint.canonical_bytes().unwrap(),
        )
        .unwrap();
    let recovery = coordinator.recovery_closure(&store).unwrap();
    assert_eq!(recovery.checkpoint, checkpoint);
    let mut wrong_preparation = preparation.clone();
    wrong_preparation.terminal_schedule_ordinals = vec![1, 4];
    assert!(
        StructuralTransientSequenceCoordinator::restore_from_closure(
            wrong_preparation,
            &recovery.root,
            &recovery.objects,
        )
        .is_err()
    );
    let mut incomplete = recovery.objects.clone();
    let removed = incomplete
        .iter()
        .position(|object| object.reference != recovery.root)
        .unwrap();
    incomplete.remove(removed);
    assert!(
        StructuralTransientSequenceCoordinator::restore_from_closure(
            preparation.clone(),
            &recovery.root,
            &incomplete,
        )
        .is_err()
    );
    let extra_object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "analysis/unrelated-transient-recovery-object",
        "application/vnd.runmat.test",
        b"unrelated".to_vec(),
    )
    .unwrap();
    let mut with_extra = recovery.objects.clone();
    with_extra.push(ProgramExecutionObject {
        reference: access()
            .reference(&extra_object, ValueRefKind::ResultObject, "runmat.test.v1")
            .unwrap(),
        bytes: extra_object.bytes,
    });
    assert!(
        StructuralTransientSequenceCoordinator::restore_from_closure(
            preparation.clone(),
            &recovery.root,
            &with_extra,
        )
        .is_err()
    );
    let mut coordinator = StructuralTransientSequenceCoordinator::restore_from_closure(
        preparation.clone(),
        &recovery.root,
        &recovery.objects,
    )
    .unwrap();
    assert_eq!(coordinator.completed_blocks().len(), 1);

    let second = coordinator.prepare_next_block().unwrap();
    let second_response = execute_durable_block(&mut store, &second);
    coordinator
        .complete_block(&store, second_response, &second)
        .unwrap();
    assert_eq!(coordinator.completed_blocks().len(), 2);
    assert!(coordinator.prepare_next_block().is_err());

    let assembly = coordinator.prepare_assembly().unwrap();
    let assembly_response = execute_durable_result(&mut store, &assembly);
    coordinator
        .complete_assembly(&store, assembly_response, &assembly)
        .unwrap();
    assert!(coordinator.is_complete());
    let restored = coordinator.finish().unwrap();
    assert_eq!(restored.result.samples().len(), 5);
    assert_eq!(restored.numerical_receipt.blocks.len(), 2);

    let uninterrupted_directory = tempfile::tempdir().unwrap();
    let mut uninterrupted_store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        uninterrupted_directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let mut uninterrupted = StructuralTransientSequenceCoordinator::new(preparation).unwrap();
    for _ in 0..2 {
        let block = uninterrupted.prepare_next_block().unwrap();
        let response = execute_durable_block(&mut uninterrupted_store, &block);
        uninterrupted
            .complete_block(&uninterrupted_store, response, &block)
            .unwrap();
    }
    let assembly = uninterrupted.prepare_assembly().unwrap();
    let response = execute_durable_result(&mut uninterrupted_store, &assembly);
    uninterrupted
        .complete_assembly(&uninterrupted_store, response, &assembly)
        .unwrap();
    let uninterrupted = uninterrupted.finish().unwrap();
    assert_eq!(restored.result, uninterrupted.result);
    assert_eq!(restored.numerical_receipt, uninterrupted.numerical_receipt);
}

#[test]
fn transient_sequence_preserves_typed_scope_resource_rejection() {
    let (mut model, mesh) = transient_fixture();
    let step_id = model.sequence.steps[0].step_id.clone();
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        solver: StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    let error =
        match prepare_structural_transient_scope_task(StructuralTransientScopeTaskPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            workload,
            mesh,
            modal_state: None,
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: ObjectInventoryLimits::default(),
        }) {
            Ok(_) => panic!("over-ceiling transient scope was prepared"),
            Err(error) => error,
        };
    assert!(matches!(
        error,
        runmat_analysis_execution::AnalysisBridgeError::FeaRun(
            runmat_analysis_fea::contracts::FeaRunError::Resource(
                runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: "resident memory bytes",
                    ..
                }
            )
        )
    ));
}

#[test]
fn modal_transient_sequence_transports_one_basis_and_restarts_exact_reduced_state() {
    for (workload, mesh, modal_state) in [modal_transient_fixture(), modal_support_motion_fixture()]
    {
        let limits = ObjectInventoryLimits::default();
        let plan =
            prepare_structural_transient_sequence(StructuralTransientSequencePreparationInput {
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                workload: &workload,
                mesh: &mesh,
                modal_state: Some(&modal_state),
                block_plan: StructuralTransientBlockPlan::new(2).unwrap(),
                initial_dependencies: Vec::new(),
                artifact_access: access(),
                program_revision: revision(),
                chunk_bytes: 256,
                inventory_limits: limits,
            })
            .unwrap();
        let scope_task =
            prepare_structural_transient_scope_task(StructuralTransientScopeTaskPreparation {
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                workload: workload.clone(),
                mesh: mesh.clone(),
                modal_state: Some(modal_state.clone()),
                artifact_access: access(),
                program_revision: revision(),
                chunk_bytes: 256,
                inventory_limits: limits,
            })
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
            directory.path().join("objects"),
            limits.max_object_bytes,
        )
        .unwrap();
        scope_task.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &scope_task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            256,
            limits,
        )
        .unwrap();
        let scope_response = AnalysisHostResponse::completed(&scope_task.host, &completed)
            .unwrap()
            .program_response();
        let imported_scope =
            import_structural_transient_scope(&store, scope_response, &scope_task, limits).unwrap();
        let workload_chunk = scope_task.inputs[0].objects[1].bytes.as_ptr();
        let mesh_chunk = scope_task.inputs[1].objects[1].bytes.as_ptr();
        let modal_chunk = scope_task.inputs[2].objects[1].bytes.as_ptr();
        let scope_identity =
            AnalysisTaskIdentity::new(scope_task.host.workload.identity, Vec::new()).unwrap();
        let preparation = complete_structural_transient_sequence_preparation(
            plan,
            scope_task,
            imported_scope,
            scope_identity,
        )
        .unwrap();
        assert_eq!(
            preparation.workload.objects[1].bytes.as_ptr(),
            workload_chunk
        );
        assert_eq!(preparation.mesh.objects[1].bytes.as_ptr(), mesh_chunk);
        assert_eq!(
            preparation.modal_state.as_ref().unwrap().objects[1]
                .bytes
                .as_ptr(),
            modal_chunk
        );
        assert_eq!(
            preparation
                .modal_state
                .as_ref()
                .unwrap()
                .manifest
                .logical_identity,
            modal_state.identity.digest()
        );

        let mut coordinator =
            StructuralTransientSequenceCoordinator::new(preparation.clone()).unwrap();
        let first = coordinator.prepare_next_block().unwrap();
        assert_eq!(first.modal_state.as_ref(), Some(&modal_state));
        assert_eq!(
            first
                .inputs
                .iter()
                .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::SemanticState)
                .count(),
            1
        );
        let response = execute_durable_block(&mut store, &first);
        coordinator
            .complete_block(&store, response, &first)
            .unwrap();
        let recovery = coordinator.recovery_closure(&store).unwrap();
        let first_terminal = coordinator.completed_blocks()[0]
            .segment
            .manifest()
            .terminal_checkpoint
            .clone();
        let mut coordinator = StructuralTransientSequenceCoordinator::restore_from_closure(
            preparation.clone(),
            &recovery.root,
            &recovery.objects,
        )
        .unwrap();
        assert_eq!(
            coordinator.completed_blocks()[0]
                .segment
                .manifest()
                .terminal_checkpoint,
            first_terminal
        );
        while coordinator.has_remaining_blocks() {
            let block = coordinator.prepare_next_block().unwrap();
            assert_eq!(block.modal_state.as_ref(), Some(&modal_state));
            let response = execute_durable_block(&mut store, &block);
            coordinator
                .complete_block(&store, response, &block)
                .unwrap();
        }
        let terminal_checkpoint = coordinator
            .completed_blocks()
            .last()
            .unwrap()
            .segment
            .manifest()
            .terminal_checkpoint
            .clone();
        let assembly = coordinator.prepare_assembly().unwrap();
        let response = execute_durable_result(&mut store, &assembly);
        coordinator
            .complete_assembly(&store, response, &assembly)
            .unwrap();
        let completed = coordinator.finish().unwrap();
        assert!(matches!(
            terminal_checkpoint.numerical_state,
            runmat_analysis_fea::execution::StructuralTransientNumericalState::ModalReduced {
                basis_state,
                ..
            } if basis_state == modal_state.identity
        ));
        assert!(completed.numerical_receipt.blocks.iter().all(|block| {
        block.solver_method
            == runmat_analysis_fea::contracts::SolverMethod::StructuralModalTransientNewmarkDenseFactorization
            && block.counters.dense_solve_count > 0
            && block.counters.dense_solve_count == block.counters.dense_factorization_count
    }));

        let mut missing_state = preparation;
        missing_state.modal_state = None;
        assert!(StructuralTransientSequenceCoordinator::new(missing_state).is_err());
    }
}

#[test]
fn authored_modal_transient_study_publishes_and_binds_the_produced_basis() {
    let (model, mesh) = authored_modal_transient_fixture();
    let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: runmat_analysis_fea::ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let template = StructuralModalTransientTemplate {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("response".into()),
        backend: ComputeBackend::Cpu,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-8,
    };
    let workloads = StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: BTreeMap::from([
            (
                ProcedureStepId("modes".into()),
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal),
                },
            ),
            (
                ProcedureStepId("response".into()),
                StructuralStepWorkload::ModalTransient {
                    template: Box::new(template),
                    block_plan: StructuralTransientBlockPlan::new(2).unwrap(),
                },
            ),
        ]),
    };
    let limits = ObjectInventoryLimits::default();
    let mut coordinator =
        StructuralModalTransientStudyCoordinator::new(StructuralModalTransientStudyPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("authored-modal-transient"),
            frozen_project_digest: None,
            workloads,
            mesh,
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
        })
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();

    let modal = coordinator.prepare_modal().unwrap();
    modal.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &modal.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&modal.host, &completed)
        .unwrap()
        .program_response();
    coordinator
        .complete_modal(&store, response, &modal)
        .unwrap();
    let modal_task = coordinator.modal_task_identity().unwrap();

    let state = coordinator.prepare_state().unwrap();
    state.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &state.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&state.host, &completed)
        .unwrap()
        .program_response();
    coordinator
        .complete_state(&store, response, &state)
        .unwrap();
    let state_task = coordinator.state_task_identity().unwrap();
    assert_eq!(
        state_task,
        runmat_analysis_fea::execution::AnalysisTaskIdentity::new(
            state.host.workload.identity,
            vec![modal_task]
        )
        .unwrap()
    );

    let plan = coordinator.prepare_transient().unwrap();
    assert_eq!(plan.initial_dependencies, vec![state_task]);
    let published = coordinator.semantic_state().unwrap();
    assert!(matches!(
        plan.workload.solver,
        StructuralTransientSolverPlan::ModalReduced { ref basis_transition }
            if basis_transition.source_state == published.state.identity
    ));

    let scope_task =
        prepare_structural_transient_scope_task(StructuralTransientScopeTaskPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            workload: plan.workload.clone(),
            mesh: plan.mesh.clone(),
            modal_state: plan.modal_state.clone(),
            artifact_access: plan.artifact_access.clone(),
            program_revision: plan.program_revision.clone(),
            chunk_bytes: plan.chunk_bytes,
            inventory_limits: plan.inventory_limits,
        })
        .unwrap();
    scope_task.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &scope_task.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let scope_response = AnalysisHostResponse::completed(&scope_task.host, &completed)
        .unwrap()
        .program_response();
    let imported_scope =
        import_structural_transient_scope(&store, scope_response, &scope_task, limits).unwrap();
    let scope_identity =
        AnalysisTaskIdentity::new(scope_task.host.workload.identity, vec![state_task]).unwrap();
    let preparation = complete_structural_transient_sequence_preparation(
        plan,
        scope_task,
        imported_scope,
        scope_identity,
    )
    .unwrap();
    let mut transient = StructuralTransientSequenceCoordinator::new(preparation).unwrap();
    let first = transient.prepare_next_block().unwrap();
    let response = execute_durable_block(&mut store, &first);
    let expected_first_task = runmat_analysis_fea::execution::AnalysisTaskIdentity::new(
        first.host.workload.identity,
        vec![state_task, scope_identity],
    )
    .unwrap();
    transient.complete_block(&store, response, &first).unwrap();
    assert_eq!(
        transient.completed_block_task_identities(),
        &[expected_first_task]
    );
}
use std::collections::{BTreeMap, BTreeSet};

#[path = "transient_block_tests/chain.rs"]
mod chain;
#[path = "transient_block_tests/field_inputs.rs"]
mod field_inputs;
