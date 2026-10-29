//! Original native solve closures feed governed adaptation without another solve.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    evaluate_original_structural_adaptation, execute_serial_analysis,
    import_structural_linear_completion, prepare_structural_linear_program, AnalysisHostResponse,
    ImportedStructuralLinearResult, PreparedStructuralLinearProgram,
    StructuralAdaptationNativeIteration, StructuralAdaptationNativeIterationInput,
    StructuralAdaptationNativeSource, StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::adaptation::{
    StructuralAdaptationSourceLimits, StructuralRecoveryEstimatorOptions,
};
use runmat_analysis_fea::solve::{
    backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{
    cache::FilesystemObjectStore, object::ObjectInventoryLimits, ArtifactAccess,
};
use runmat_meshing_core::*;
mod cold;
mod compilation_budget;
mod fixture;
mod process;

fn prepare(model: &AnalysisModel, mesh: &SolverMeshArtifact) -> PreparedStructuralLinearProgram {
    let digest = Digest::sha256(b"original-native-adaptive-patch");
    prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            digest,
            digest,
            ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "study:adaptive-patch".into(),
            encryption_context: digest,
        },
        workload: StructuralLinearWorkload {
            schema_version: 2,
            model: model.clone(),
            step_id: model.sequence.steps[0].step_id.clone(),
            backend: ComputeBackend::Cpu,
            preconditioner: SpdPreconditionerKind::Ilu0,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
            require_solver_mesh_for_solid: true,
            state_input: None,
        },
        mesh: Some(mesh.clone()),
        state_input: None,
        chunk_bytes: 1024,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: Default::default(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap()
}

fn produce(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    store: &mut FilesystemObjectStore,
) -> (
    PreparedStructuralLinearProgram,
    ImportedStructuralLinearResult,
) {
    let prepared = prepare(model, mesh);
    prepared.program.write_to(store).unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        store,
        &NativeAnalysisKernel,
        1024,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response();
    let result = import_structural_linear_completion(
        store,
        response,
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    (prepared, result)
}

fn original<'a>(
    prepared: &'a PreparedStructuralLinearProgram,
    result: &'a ImportedStructuralLinearResult,
) -> StructuralAdaptationNativeSource<'a> {
    StructuralAdaptationNativeSource {
        host: &prepared.host,
        invocation: prepared
            .inputs
            .iter()
            .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
            .unwrap(),
        mesh: prepared
            .inputs
            .iter()
            .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::Mesh)
            .unwrap(),
        result: &result.objects,
        receipt: &result.numerical_receipt_objects,
    }
}

#[test]
fn original_native_h_refinement_authenticates_receipts_and_cold_adaptive_evidence() {
    let coarse = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let fine = fixtures::connected_tetrahedron_solver_mesh();
    let mut model = fixture::model(&coarse);
    let accuracy = fixture::plan();
    model.sequence.steps[0].accuracy = accuracy.clone();
    let (map, lineage) = fixture::lineage(&coarse, &fine);
    let directory = tempfile::tempdir().unwrap();
    let mut store = FilesystemObjectStore::open(
        directory.path().join("objects"),
        ObjectInventoryLimits::default().max_object_bytes,
    )
    .unwrap();
    let (source, source_result) = produce(&model, &coarse, &mut store);
    let (target, target_result) = produce(&model, &fine, &mut store);
    assert!(
        target_result
            .result
            .run
            .structural_static_convergence
            .as_ref()
            .unwrap()
            .iterations
            > 0
    );
    fixture::assert_affine_original(&coarse, &source_result.result.run);
    fixture::assert_affine_original(&fine, &target_result.result.run);
    let id = ObservationId("tip_displacement".into());
    let mut input = StructuralAdaptationNativeIterationInput {
        source: original(&source, &source_result),
        target: original(&target, &target_result),
        transfer_map: &map,
        lineage: &lineage,
        observation_id: &id,
        accuracy: &accuracy,
        source_limits: fixture::source_limits(),
        maximum_source_decode_bytes: 256 * 1024 * 1024,
        estimator_options: StructuralRecoveryEstimatorOptions::default(),
        previous: None,
    };
    let originals = [input.source, input.target]
        .into_iter()
        .flat_map(|s| [s.invocation, s.mesh, s.result, s.receipt]);
    let required = runmat_analysis_execution::prepared_input_resource_facts_from_refs(originals)
        .unwrap()
        .decoded_resident_bytes_admission_budget
        .0;
    input.maximum_source_decode_bytes = required;
    let evidence = evaluate_original_structural_adaptation(input).unwrap();
    assert_eq!(evidence.target_result, target_result.identity);
    assert_eq!(
        evidence.target_receipt,
        target_result.numerical_receipt.identity()
    );
    assert_eq!(
        evidence.solved.iteration.decision.status,
        runmat_analysis_fea::adaptation::StructuralAdaptationDecisionStatus::Converged
    );
    let cold: StructuralAdaptationNativeIteration =
        serde_json::from_slice(&serde_json::to_vec(&evidence).unwrap()).unwrap();
    cold.validate_against(input).unwrap();
    let mut changed = cold.clone();
    changed.target_receipt = source_result.numerical_receipt.identity();
    assert!(changed.validate_against(input).is_err());
    let wrong_receipt = StructuralAdaptationNativeSource {
        receipt: &source_result.numerical_receipt_objects,
        ..input.target
    };
    assert!(
        evaluate_original_structural_adaptation(StructuralAdaptationNativeIterationInput {
            target: wrong_receipt,
            ..input
        })
        .is_err()
    );
    assert!(
        evaluate_original_structural_adaptation(StructuralAdaptationNativeIterationInput {
            maximum_source_decode_bytes: required - 1,
            ..input
        })
        .is_err()
    );
    assert!(
        evaluate_original_structural_adaptation(StructuralAdaptationNativeIterationInput {
            previous: Some(&cold),
            ..input
        })
        .is_err()
    );
    let mut wrong_access = target.host.clone();
    wrong_access.artifact_access.authorization_scope = "study:foreign".into();
    assert!(
        evaluate_original_structural_adaptation(StructuralAdaptationNativeIterationInput {
            target: StructuralAdaptationNativeSource {
                host: &wrong_access,
                ..input.target
            },
            ..input
        })
        .is_err()
    );
    drop(store);
    let reopened = FilesystemObjectStore::open(
        directory.path().join("objects"),
        ObjectInventoryLimits::default().max_object_bytes,
    )
    .unwrap();
    let source_cold = cold::reload(&reopened, input.source);
    let target_cold = cold::reload(&reopened, input.target);
    cold.validate_against(StructuralAdaptationNativeIterationInput {
        source: source_cold.source(&source.host),
        target: target_cold.source(&target.host),
        ..input
    })
    .unwrap();
}
