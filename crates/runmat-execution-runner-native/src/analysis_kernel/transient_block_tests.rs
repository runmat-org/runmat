use runmat_analysis_core::{
    AnalysisConstraint, ExcitationTarget, FormulationRequirement, IndependentDomain,
    PrescribedStructuralDofs, ProcedureStep, ProcedureStepId, ReferenceStatePolicy, RegionTarget,
    SequenceDependency, SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    StructuralConstraint, StructuralConstraintKind, TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_transient_block, import_structural_transient_result,
    prepare_analysis_objects, prepare_structural_transient_block_request_objects,
    prepare_structural_transient_block_task, prepare_structural_transient_checkpoint_objects,
    prepare_structural_transient_result_task, prepare_structural_transient_sequence,
    prepare_structural_transient_workload_objects, AnalysisHostResponse,
    PreparedStructuralTransientBlockTask, PreparedStructuralTransientResultTask,
    StructuralTransientBlockPreparation, StructuralTransientResultPreparation,
    StructuralTransientSequenceCoordinator, StructuralTransientSequencePreparation,
    StructuralTransientSequencePreparationInput,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, CanonicalSemanticState, CanonicalSemanticStateInput,
        CanonicalSemanticStatePayload, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralTransientBlockPlan, StructuralTransientBlockRequest,
        StructuralTransientSolverPlan, StructuralTransientWorkload,
    },
    fixtures::{fixture_model, FixtureId},
    solve::transient::StructuralTimeStepPolicy,
    ComputeBackend, ModalSolveOptions, ModalTruncationPolicy,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
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
        ProgramEnvironment::new(
            1,
            1,
            digest("transient-block-runtime"),
            digest("transient-block-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn transient_fixture() -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.material_assignments[0].region_id = "region".into();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "fixed-face".into(),
        frame: runmat_analysis_core::CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary("face:0"),
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
    model.excitations[0].excitation_mut().target =
        ExcitationTarget::Region(RegionTarget::boundary("face:3"));
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
    let (mut model, mesh) = transient_fixture();
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut transient_step =
        ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
    transient_step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
    };
    model.sequence.steps = vec![modal_step, transient_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("response".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-transient".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("response".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let modal = runmat_analysis_fea::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
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
    let workload = StructuralTransientWorkload {
        schema_version: 1,
        model,
        step_id: ProcedureStepId("response".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        solver: StructuralTransientSolverPlan::ModalReduced {
            basis_state: state.identity,
        },
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-8,
    };
    (workload, mesh, state)
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
fn native_kernel_executes_a_two_block_physical_transient_chain() {
    let (model, mesh) = transient_fixture();
    let step = &model.sequence.steps[0];
    let workload = StructuralTransientWorkload {
        schema_version: 1,
        model: model.clone(),
        step_id: step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        solver: runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload, &mesh, None,
    )
    .unwrap();
    let limits = ObjectInventoryLimits::default();
    let namespace = ObjectNamespace::InputValue;
    let workload_objects = prepare_structural_transient_workload_objects(
        &workload,
        scope.source_workload,
        256,
        namespace,
        limits,
    )
    .unwrap();
    let mesh_bytes = mesh.canonical_encode().unwrap();
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh_bytes,
        256,
        namespace,
        limits,
    )
    .unwrap();
    let first_request = StructuralTransientBlockRequest::new(scope.clone(), None, 2).unwrap();
    let first_request_objects =
        prepare_structural_transient_block_request_objects(&first_request, 256, namespace, limits)
            .unwrap();
    let first = prepare_structural_transient_block_task(StructuralTransientBlockPreparation {
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        request: first_request_objects,
        predecessor: None,
        modal_state: None,
        artifact_access: access(),
        program_revision: revision(),
    })
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let store_root = directory.path().join("objects");
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let first_response = execute_durable_block(&mut store, &first);
    drop(store);

    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let first_import =
        import_structural_transient_block(&store, first_response, &first, limits).unwrap();
    let checkpoint = first_import.segment.manifest().terminal_checkpoint.clone();
    let second_request =
        StructuralTransientBlockRequest::new(scope.clone(), Some(&checkpoint), 4).unwrap();
    let second_request_objects =
        prepare_structural_transient_block_request_objects(&second_request, 256, namespace, limits)
            .unwrap();
    let checkpoint_objects = prepare_structural_transient_checkpoint_objects(
        &checkpoint,
        256,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    let second = prepare_structural_transient_block_task(StructuralTransientBlockPreparation {
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        request: second_request_objects,
        predecessor: Some(checkpoint_objects),
        modal_state: None,
        artifact_access: access(),
        program_revision: revision(),
    })
    .unwrap();
    let second_response = execute_durable_block(&mut store, &second);
    drop(store);

    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let second_import =
        import_structural_transient_block(&store, second_response, &second, limits).unwrap();

    assert_eq!(first_import.segment.samples().len(), 2);
    assert_eq!(second_import.segment.samples().len(), 2);
    assert_eq!(
        second_import
            .segment
            .manifest()
            .terminal_checkpoint
            .continuation
            .schedule_ordinal,
        4
    );
    assert_eq!(
        second_import
            .segment
            .manifest()
            .terminal_checkpoint
            .predecessor,
        Some(checkpoint.identity)
    );

    let result_task =
        prepare_structural_transient_result_task(StructuralTransientResultPreparation {
            blocks: vec![second_import, first_import],
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let result_response = execute_durable_result(&mut store, &result_task);
    drop(store);

    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let imported =
        import_structural_transient_result(&store, result_response, &result_task, limits).unwrap();
    assert_eq!(imported.result.samples().len(), 5);
    assert_eq!(
        imported.result.samples()[0].schedule_ordinal,
        0,
        "the first computed physical state must survive block partitioning"
    );
    assert_eq!(imported.numerical_receipt.blocks.len(), 2);
    assert_eq!(
        imported.numerical_receipt.source_result,
        imported.result.manifest().identity
    );

    let one_request = StructuralTransientBlockRequest::new(scope, None, 4).unwrap();
    let one_request_objects =
        prepare_structural_transient_block_request_objects(&one_request, 256, namespace, limits)
            .unwrap();
    let one_block = prepare_structural_transient_block_task(StructuralTransientBlockPreparation {
        workload: workload_objects,
        mesh: mesh_objects,
        request: one_request_objects,
        predecessor: None,
        modal_state: None,
        artifact_access: access(),
        program_revision: revision(),
    })
    .unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_response = execute_durable_block(&mut store, &one_block);
    drop(store);
    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_import =
        import_structural_transient_block(&store, one_response, &one_block, limits).unwrap();
    let one_result_task =
        prepare_structural_transient_result_task(StructuralTransientResultPreparation {
            blocks: vec![one_import],
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_result_response = execute_durable_result(&mut store, &one_result_task);
    drop(store);
    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_result =
        import_structural_transient_result(&store, one_result_response, &one_result_task, limits)
            .unwrap();

    assert_eq!(one_result.result, imported.result);
    assert_ne!(
        one_result.numerical_receipt.identity, imported.numerical_receipt.identity,
        "execution partitioning belongs to the receipt, not the physical result"
    );
}

#[test]
fn transient_sequence_coordinator_admits_each_exact_transition_before_assembly() {
    let (model, mesh) = transient_fixture();
    let step_id = model.sequence.steps[0].step_id.clone();
    let workload = StructuralTransientWorkload {
        schema_version: 1,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
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
        workload: workload_objects,
        mesh: mesh_objects,
        modal_state: None,
        scope,
        terminal_schedule_ordinals: vec![2, 4],
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
fn modal_transient_sequence_transports_one_basis_and_restarts_exact_reduced_state() {
    let (workload, mesh, modal_state) = modal_transient_fixture();
    let limits = ObjectInventoryLimits::default();
    let preparation =
        prepare_structural_transient_sequence(StructuralTransientSequencePreparationInput {
            workload: &workload,
            mesh: &mesh,
            modal_state: Some(&modal_state),
            block_plan: StructuralTransientBlockPlan::new(2).unwrap(),
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    assert_eq!(
        preparation
            .modal_state
            .as_ref()
            .unwrap()
            .manifest
            .logical_identity,
        modal_state.identity.digest()
    );

    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let mut coordinator = StructuralTransientSequenceCoordinator::new(preparation.clone()).unwrap();
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
