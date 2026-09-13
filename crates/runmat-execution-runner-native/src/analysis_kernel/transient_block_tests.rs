use runmat_analysis_core::{
    AnalysisConstraint, ExcitationTarget, FormulationRequirement, PrescribedStructuralDofs,
    RegionTarget, StructuralConstraint, StructuralConstraintKind,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_transient_block, prepare_analysis_objects,
    prepare_structural_transient_block_request_objects, prepare_structural_transient_block_task,
    prepare_structural_transient_checkpoint_objects, prepare_structural_transient_workload_objects,
    AnalysisHostResponse, PreparedStructuralTransientBlockTask,
    StructuralTransientBlockPreparation,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, StructuralTransientBlockRequest, StructuralTransientWorkload,
    },
    fixtures::{fixture_model, FixtureId},
    solve::transient::StructuralTimeStepPolicy,
    ComputeBackend,
};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
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
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload, &mesh,
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
    let second_request = StructuralTransientBlockRequest::new(scope, Some(&checkpoint), 4).unwrap();
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
        workload: workload_objects,
        mesh: mesh_objects,
        request: second_request_objects,
        predecessor: Some(checkpoint_objects),
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
    let second_segment = second_import.segment;

    assert_eq!(first_import.segment.samples().len(), 2);
    assert_eq!(second_segment.samples().len(), 2);
    assert_eq!(
        second_segment
            .manifest()
            .terminal_checkpoint
            .continuation
            .schedule_ordinal,
        4
    );
    assert_eq!(
        second_segment.manifest().terminal_checkpoint.predecessor,
        Some(checkpoint.identity)
    );
}
