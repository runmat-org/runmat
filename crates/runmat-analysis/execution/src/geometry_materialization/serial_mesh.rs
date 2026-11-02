//! Serial composition of the production geometry and meshing owners.
use super::{
    execute_geometry_materialization_program_request, import_geometry_materialization_output,
    GeometryMaterializationLimits, GeometryMaterializationOutput, OcctMeshingEvaluatorProvider,
};
use runmat_execution::value::ValuePayload;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    object::ObjectInventoryLimits,
    ArtifactAccess, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_meshing_core::{
    CacheAdmissionDecision, MeshingCancellationSignal, MeshingChunkPolicy, MeshingRequestSettings,
    PlatformBuildIdentity,
};
use runmat_meshing_execution::{
    import_exact_geometry_input, ExactMeshingDagRun, ExactMeshingDagRunResult,
    MeshingKernelDispatcher, MeshingProgressSink, MeshingRunEvidenceContext,
    SerialExactMeshingExecutor,
};

pub struct SerialGeometryMeshingResult {
    pub geometry: GeometryMaterializationOutput,
    pub meshing: ExactMeshingDagRunResult,
}

/// Keep the mesher's canonical diagnostic intact through host adapters.
#[derive(Debug, thiserror::Error)]
pub enum SerialGeometryMeshingError {
    #[error("{0}")]
    Geometry(String),
    #[error("{0}")]
    Meshing(#[from] runmat_meshing_execution::ExactMeshingDagRunError),
}
impl From<String> for SerialGeometryMeshingError {
    fn from(value: String) -> Self {
        Self::Geometry(value)
    }
}
impl From<&str> for SerialGeometryMeshingError {
    fn from(value: &str) -> Self {
        Self::Geometry(value.into())
    }
}
impl SerialGeometryMeshingError {
    pub fn meshing_failure(&self) -> Option<&runmat_meshing_core::MeshingFailure> {
        match self {
            Self::Meshing(runmat_meshing_execution::ExactMeshingDagRunError::Stage(error)) => {
                error.failure()
            }
            _ => None,
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "FEA alpha execution-context structure is deferred under the approved 1.0 hardening scope"
)]
pub fn materialize_and_mesh_serial<S: CacheImport + CacheExport>(
    program: &ProgramExecutionRequest,
    store: &mut S,
    settings: MeshingRequestSettings,
    ceilings: runmat_meshing_core::MeshingExecutionCeilings,
    platform: PlatformBuildIdentity,
    inventory: ObjectInventoryLimits,
    cancellation: &dyn MeshingCancellationSignal,
    progress: &mut dyn MeshingProgressSink,
) -> Result<SerialGeometryMeshingResult, SerialGeometryMeshingError> {
    let descriptor = match program.arguments.first() {
        Some(ValuePayload::Object(value)) => value,
        _ => return Err("Geometry preparation has no admitted descriptor.".into()),
    };
    let access = ArtifactAccess {
        authorization_scope: descriptor.authorization_scope.clone(),
        encryption_context: descriptor.encryption_context,
    };
    let response = execute_geometry_materialization_program_request(
        program,
        store,
        GeometryMaterializationLimits { inventory },
        platform.clone(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        return Err(format!("Geometry preparation failed: {response:?}").into());
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err("Geometry preparation returned an invalid root.".into());
    };
    let output =
        import_geometry_materialization_output(store, root, &access).map_err(|e| e.to_string())?;
    let geometry = import_exact_geometry_input(
        store,
        output.document.clone(),
        &output.geometry_root,
        access.clone(),
        inventory,
    )
    .map_err(|e| e.to_string())?;
    let request = runmat_meshing_core::resolve_meshing_request(output.document.tolerance, settings)
        .map_err(|e| e.to_string())?;
    let kernel = MeshingKernelDispatcher::new(OcctMeshingEvaluatorProvider);
    let mut executor = SerialExactMeshingExecutor {
        store,
        kernel: &kernel,
        cancellation,
        progress,
        chunk_policy: MeshingChunkPolicy {
            maximum_chunk_bytes: 8 << 20,
            maximum_records_per_chunk: 65_536,
            maximum_total_encoded_bytes: inventory.max_total_bytes,
        },
        inventory_limits: inventory,
    };
    let meshing = runmat_meshing_execution::execute_exact_meshing_dag(
        ExactMeshingDagRun {
            geometry: &geometry,
            request,
            ceilings,
            artifact_access: access,
            capability_cohort: Some(platform.capability_cohort.clone()),
            program_revision: program.recipe.program_revision.clone(),
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
            inventory_limits: inventory,
            evidence: MeshingRunEvidenceContext {
                platform,
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
            named_regions: runmat_meshing_core::NamedRegionEntities::from_resolved_regions(
                &output.regions,
            ),
        },
        &mut executor,
    )?;
    Ok(SerialGeometryMeshingResult {
        geometry: output,
        meshing,
    })
}

/// Validated geometry/mesh product transferred between execution hosts.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PreparedFeaGeometry {
    pub geometry: GeometryMaterializationOutput,
    pub mesh: runmat_meshing_core::SolverMeshArtifact,
    pub meshing_evidence: runmat_meshing_core::MeshingEvidence,
}
