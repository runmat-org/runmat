//! Local composition of the same portable whole-study root used by remote execution.

use runmat_analysis_fea::study::{StructuralStudyProduct, StructuralStudyProductInput};
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_meshing_core::CacheAdmissionDecision;
use runmat_meshing_execution::{
    execute_exact_meshing_dag_async, import_exact_geometry_input, prepare_exact_geometry_input,
    ExactMeshingDagRun, ExactMeshingDagRunResult, MeshingRunEvidenceContext,
};

use crate::{
    execute_structural_workloads_locally, NativeExactMeshingExecutor, NativeExecutionConfig,
    NativeExecutionError, NativeExecutionResult, NativeGeometryMaterializationLimits,
    NativeProgramSession, PreparedStructuralStudyRoot,
};

/// Complete local product of a typed whole-study root.
pub struct CompletedLocalStructuralStudy {
    pub product: StructuralStudyProduct,
    pub meshing: ExactMeshingDagRunResult,
    pub numerical: runmat_runtime::analysis::StructuralStudyNumericalResult,
}

/// Execute a portable typed study through geometry, exact meshing, compilation, and solve.
///
/// The input is the same immutable root transported to a remote driver. Local execution only
/// supplies process placement and storage; it does not reconstruct scientific intent.
pub async fn execute_prepared_structural_study_locally(
    mut config: NativeExecutionConfig,
    prepared: PreparedStructuralStudyRoot,
    workers: u32,
) -> NativeExecutionResult<CompletedLocalStructuralStudy> {
    let request = prepared.request().clone();
    let inventory = ObjectInventoryLimits::default();
    let composition_root = config
        .store_root
        .join("whole-studies")
        .join(uuid::Uuid::new_v4().simple().to_string());
    let mut composition_store =
        crate::FilesystemObjectStore::open(composition_root, inventory.max_object_bytes)
            .map_err(protocol)?;
    prepared.write_to(&mut composition_store)?;

    let decoded = crate::decode_structural_study_root(&request, &composition_store)?;
    let geometry_meshing =
        crate::decode_geometry_meshing_root(&decoded.geometry_meshing, &composition_store)?;
    let (geometry_request, _) = runmat_analysis_execution::decode_geometry_materialization_program(
        &geometry_meshing.geometry,
        &composition_store,
        &decoded.artifact_access,
    )
    .map_err(protocol)?;
    let response = crate::execute_geometry_materialization_program_request(
        &geometry_meshing.geometry,
        &mut composition_store,
        NativeGeometryMaterializationLimits { inventory },
    );
    let output_root = materialization_output_root(response)?;
    let output = runmat_analysis_execution::import_geometry_materialization_output(
        &composition_store,
        &output_root,
        &decoded.artifact_access,
    )
    .map_err(protocol)?;
    output
        .validate_against(&geometry_request)
        .map_err(protocol)?;

    let exact = import_exact_geometry_input(
        &composition_store,
        output.document.clone(),
        &output.geometry_root,
        decoded.artifact_access.clone(),
        inventory,
    )
    .map_err(protocol)?;
    let mut meshing_request = geometry_meshing.request.meshing_request;
    if !crate::exact_geometry_admission::same_requested_tolerance(
        &meshing_request.tolerance,
        &output.document.tolerance,
    ) {
        return Err(protocol(
            "meshing request tolerance differs from materialized exact geometry",
        ));
    }
    meshing_request.tolerance = output.document.tolerance;

    let analysis_config = config.clone();
    config.enable_exact_meshing(
        &output.document,
        &meshing_request,
        Some(&output.platform.capability_cohort),
    )?;
    let session = NativeProgramSession::new(config)?;
    let mut session_store = session.object_store();
    for object in &exact.geometry_objects().objects {
        session_store.write_verified(object).map_err(protocol)?;
    }
    let geometry = prepare_exact_geometry_input(
        exact.geometry_objects().clone(),
        decoded.artifact_access.clone(),
        inventory,
    )
    .map_err(protocol)?;
    let mut executor =
        NativeExactMeshingExecutor::new(&session, crate::MeshingExecutionPolicy::default())
            .map_err(protocol)?;
    let meshing = execute_exact_meshing_dag_async(
        ExactMeshingDagRun {
            geometry: &geometry,
            request: meshing_request,
            artifact_access: decoded.artifact_access,
            capability_cohort: Some(output.platform.capability_cohort.clone()),
            program_revision: request.recipe.program_revision.clone(),
            preferred_edges_per_partition: geometry_meshing.request.preferred_edges_per_partition,
            preferred_faces_per_partition: geometry_meshing.request.preferred_faces_per_partition,
            inventory_limits: inventory,
            evidence: MeshingRunEvidenceContext {
                platform: output.platform,
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
        },
        &mut executor,
    )
    .await
    .map_err(protocol)?;

    let product =
        runmat_analysis_fea::study::build_structural_study_product(StructuralStudyProductInput {
            study: &decoded.study,
            geometry: &output.document,
            regions: &output.regions,
            materials: &decoded.materials,
            mesh: &meshing.artifact,
            planning: decoded.planning,
        })
        .map_err(protocol)?;
    let numerical = execute_structural_workloads_locally(crate::StructuralWorkloadLocalExecution {
        execution_config: analysis_config,
        program_environment: request.recipe.program_revision.environment(),
        study_digest: decoded.authored_study_digest,
        frozen_project_digest: decoded.frozen_project_digest,
        workloads: product.workloads.clone(),
        mesh: meshing.artifact.clone(),
        workers,
        admitted_accelerators: Some(&decoded.analysis_accelerator_requests),
    })
    .await?;
    Ok(CompletedLocalStructuralStudy {
        product,
        meshing,
        numerical,
    })
}

fn materialization_output_root(
    response: ProgramExecutionResponse,
) -> NativeExecutionResult<runmat_execution::value::ValueRef> {
    match response {
        ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } => match outputs.as_slice() {
            [ValuePayload::Object(reference)] if reference.kind == ValueRefKind::ResultObject => {
                Ok(reference.as_ref().clone())
            }
            _ => Err(protocol(
                "geometry materialization returned an invalid output root",
            )),
        },
        ProgramExecutionResponse::Failure { message } => Err(protocol(message)),
        ProgramExecutionResponse::Success { .. } => Err(protocol(
            "geometry materialization returned an inline output",
        )),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
