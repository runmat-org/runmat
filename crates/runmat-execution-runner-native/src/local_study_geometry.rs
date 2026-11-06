//! Shared local exact-geometry and meshing authority for typed study families.
use crate::local_structural_study::emit_local_progress;
use crate::{
    FilesystemObjectStore, LocalStructuralStudyObservers, NativeExactMeshingExecutor,
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult,
    NativeGeometryMaterializationLimits, NativeProgramSession,
};
use runmat_analysis_execution::GeometryMaterializationOutput;
use runmat_analysis_fea::{FeaProgressPhase, FeaProgressStatus};
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution_artifact::{
    cache::CacheExport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionRequest,
    ProgramExecutionResponse,
};
use runmat_meshing_core::CacheAdmissionDecision;
use runmat_meshing_execution::{
    execute_exact_meshing_dag_async, import_exact_geometry_input, prepare_exact_geometry_input,
    ExactMeshingDagRun, ExactMeshingDagRunResult, MeshingRunEvidenceContext,
};

pub(crate) struct LocalStudyGeometry {
    pub output: GeometryMaterializationOutput,
    pub meshing: ExactMeshingDagRunResult,
    pub analysis_config: NativeExecutionConfig,
}

/// Caller emits GeometryLoad Started before opening/decoding its composition root.
pub(crate) async fn execute_local_study_geometry(
    mut config: NativeExecutionConfig,
    geometry_meshing_program: &ProgramExecutionRequest,
    artifact_access: &ArtifactAccess,
    composition_store: &mut FilesystemObjectStore,
    observers: &LocalStructuralStudyObservers,
) -> NativeExecutionResult<LocalStudyGeometry> {
    const OPERATION: &str = "fea.typed_study";
    let inventory = ObjectInventoryLimits::default();
    let geometry_meshing =
        crate::decode_geometry_meshing_root(geometry_meshing_program, composition_store)?;
    if &geometry_meshing.artifact_access != artifact_access {
        return Err(protocol(
            "geometry/meshing root differs from study artifact access",
        ));
    }
    let (geometry_request, _) = runmat_analysis_execution::decode_geometry_materialization_program(
        &geometry_meshing.geometry,
        composition_store,
        artifact_access,
    )
    .map_err(protocol)?;
    let response = crate::execute_geometry_materialization_program_request(
        &geometry_meshing.geometry,
        composition_store,
        NativeGeometryMaterializationLimits { inventory },
    );
    let output_root = materialization_output_root(response)?;
    let output = runmat_analysis_execution::import_geometry_materialization_output(
        composition_store,
        &output_root,
        artifact_access,
    )
    .map_err(protocol)?;
    output
        .validate_against(&geometry_request)
        .map_err(protocol)?;
    emit_local_progress(
        observers,
        OPERATION,
        FeaProgressPhase::GeometryLoad,
        FeaProgressStatus::Completed,
        "Study geometry loaded",
        Some(1),
        Some(1),
    );
    emit_local_progress(
        observers,
        OPERATION,
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "Geometry regions resolved against admitted topology",
        Some(output.regions.len() as u64),
        Some(output.regions.len() as u64),
    );

    emit_local_progress(
        observers,
        OPERATION,
        FeaProgressPhase::MeshPrep,
        FeaProgressStatus::Started,
        "Generating the analysis mesh",
        None,
        None,
    );
    let exact = import_exact_geometry_input(
        composition_store,
        output.document.clone(),
        &output.geometry_root,
        artifact_access.clone(),
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
    let meshing_ceilings = match config.local_memory_capacity_bytes {
        Some(capacity) => crate::machine_capacity::meshing_execution_ceilings_for(
            capacity.min(config.authored_worker_memory_bytes.unwrap_or(u64::MAX)),
            &meshing_request.limits,
        ),
        None => crate::machine_capacity::meshing_execution_ceilings(&meshing_request.limits),
    };
    // Worker slots advertise the whole machine, while the pool has one aggregate
    // capacity. A wide volume task and narrow entity tasks draw from the same budget.
    config.worker_memory_bytes = config
        .worker_memory_bytes
        .max(meshing_ceilings.maximum_memory_bytes);
    config.worker_scratch_bytes = config
        .worker_scratch_bytes
        .max(meshing_ceilings.maximum_scratch_bytes);
    let meshing_cpu_millicores = config.worker_cpu_millicores;
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
        exact.into_geometry_objects(),
        artifact_access.clone(),
        inventory,
    )
    .map_err(protocol)?;
    let mut executor = NativeExactMeshingExecutor::new(
        &session,
        crate::MeshingExecutionPolicy {
            cpu_millicores: meshing_cpu_millicores,
            ..Default::default()
        },
    )
    .map_err(protocol)?;
    let meshing = execute_exact_meshing_dag_async(
        ExactMeshingDagRun {
            geometry: &geometry,
            request: meshing_request,
            ceilings: meshing_ceilings,
            artifact_access: artifact_access.clone(),
            capability_cohort: Some(output.platform.capability_cohort.clone()),
            program_revision: geometry_meshing_program.recipe.program_revision.clone(),
            preferred_edges_per_partition: geometry_meshing.request.preferred_edges_per_partition,
            preferred_faces_per_partition: geometry_meshing.request.preferred_faces_per_partition,
            inventory_limits: inventory,
            evidence: MeshingRunEvidenceContext {
                platform: output.platform.clone(),
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
            named_regions: runmat_meshing_core::NamedRegionEntities::from_resolved_regions(
                &output.regions,
            ),
        },
        &mut executor,
    )
    .await
    .map_err(protocol)?;
    if let Some(observer) = observers.mesh.as_ref() {
        observer(&meshing.artifact);
    }
    emit_local_progress(
        observers,
        OPERATION,
        FeaProgressPhase::MeshPrep,
        FeaProgressStatus::Completed,
        format!(
            "Analysis mesh ready: {} nodes, {} elements",
            meshing.artifact.topology.nodes.len(),
            meshing.artifact.topology.volume_elements.len()
        ),
        Some(meshing.artifact.topology.volume_elements.len() as u64),
        Some(meshing.artifact.topology.volume_elements.len() as u64),
    );

    Ok(LocalStudyGeometry {
        output,
        meshing,
        analysis_config,
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
        ProgramExecutionResponse::RuntimeFailure { failure } => Err(protocol(failure.message)),
        ProgramExecutionResponse::Success { .. }
        | ProgramExecutionResponse::GraphSuccess { .. }
        | ProgramExecutionResponse::SpmdSuccess { .. } => Err(protocol(
            "geometry materialization returned an inline output",
        )),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
