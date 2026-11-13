//! Backend-neutral orchestration of the meshing-owned exact geometry DAG.

mod stage;
mod surface;
mod terminal;

use runmat_time::Instant;

use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_meshing_core::{
    CacheAdmissionDecision, CanonicalMeshingContract, MeshingChunkMediaType, MeshingEvidence,
    MeshingFailure, MeshingFailureCategory, MeshingOperation, MeshingRequest, MeshingStageEvidence,
    MeshingStageKind, PlatformBuildIdentity, SizingResolutionEvidence, SolverMeshArtifact,
    MESHING_FAILURE_SCHEMA_VERSION,
};

use crate::{ArtifactAccess, ExactMeshingDagPlanner, PreparedExactGeometryInput};

use stage::{admit_stage_success, execute_planned_stage};
pub use stage::{ExactMeshingDagExecutor, MeshingStageExecutionError, SerialExactMeshingExecutor};

#[derive(Clone, Debug, PartialEq)]
pub struct MeshingRunEvidenceContext {
    pub platform: PlatformBuildIdentity,
    pub sizing: Vec<SizingResolutionEvidence>,
    pub cache_admission: CacheAdmissionDecision,
}

pub struct ExactMeshingDagRun<'a> {
    pub geometry: &'a PreparedExactGeometryInput,
    pub request: MeshingRequest,
    /// Execution ceilings for this run (outside the request identity).
    pub ceilings: runmat_meshing_core::MeshingExecutionCeilings,
    pub artifact_access: ArtifactAccess,
    pub capability_cohort: Option<String>,
    pub program_revision: ProgramRevision,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
    pub inventory_limits: ObjectInventoryLimits,
    pub evidence: MeshingRunEvidenceContext,
    /// Named solid, face and edge regions of the materialized geometry. The request's size
    /// controls resolve against them before any stage runs; empty when the request has none.
    pub named_regions: std::collections::BTreeMap<String, runmat_meshing_core::NamedRegionEntities>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactMeshingDagRunResult {
    pub root: runmat_execution::value::ValueRef,
    pub result_objects: Vec<runmat_execution::value::ValueRef>,
    pub artifact: SolverMeshArtifact,
    pub evidence: MeshingEvidence,
}

pub fn import_exact_meshing_result(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    root: runmat_execution::value::ValueRef,
    result_objects: Vec<runmat_execution::value::ValueRef>,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> Result<ExactMeshingDagRunResult, ExactMeshingDagRunError> {
    let publication = crate::import_result_publication(source, &root, access.clone(), limits)?;
    if publication.result_objects().len().checked_add(1) != Some(result_objects.len())
        || publication
            .result_objects()
            .iter()
            .any(|reference| !result_objects.contains(reference))
    {
        return Err(ExactMeshingDagRunError::Invalid(
            "terminal result inventory differs from its canonical publication".into(),
        ));
    }
    let observations = result_objects
        .iter()
        .filter(|reference| {
            reference.media_type == crate::STAGE_EVIDENCE_MEDIA_TYPE
                || reference.value_schema == crate::STAGE_EVIDENCE_VALUE_SCHEMA
        })
        .collect::<Vec<_>>();
    let [observation] = observations.as_slice() else {
        return Err(ExactMeshingDagRunError::Invalid(
            "terminal result requires exactly one publication-stage observation".into(),
        ));
    };
    access
        .validate_reference(
            observation,
            runmat_execution::value::ValueRefKind::ResultObject,
            crate::STAGE_EVIDENCE_MEDIA_TYPE,
            crate::STAGE_EVIDENCE_VALUE_SCHEMA,
        )
        .map_err(crate::MeshingExecutionError::from)?;
    let observation_bytes = source
        .read_verified(observation.logical_digest)
        .map_err(crate::MeshingExecutionError::from)?
        .ok_or_else(|| {
            ExactMeshingDagRunError::Invalid(
                "terminal publication-stage observation is unavailable".into(),
            )
        })?;
    if observation.encoded_length != observation_bytes.len() as u64
        || observation.logical_digest != runmat_execution::Digest::sha256(&observation_bytes)
    {
        return Err(ExactMeshingDagRunError::Invalid(
            "terminal publication-stage observation differs from its reference".into(),
        ));
    }
    let observation = MeshingStageEvidence::canonical_decode(&observation_bytes)?;
    observation.validate()?;
    if observation.stage != MeshingStageKind::Publication
        || observation.stage_result_digest.bytes() != root.logical_digest.bytes()
    {
        return Err(ExactMeshingDagRunError::Invalid(
            "terminal stage observation does not bind the publication root".into(),
        ));
    }
    let streams = publication.stage_objects().decoded_streams()?;
    let [artifact_stream, evidence_stream] = streams.as_slice() else {
        return Err(ExactMeshingDagRunError::Invalid(
            "terminal publication has an invalid stream shape".into(),
        ));
    };
    let ([artifact_bytes], [evidence_bytes]) = (
        artifact_stream.records.as_slice(),
        evidence_stream.records.as_slice(),
    ) else {
        return Err(ExactMeshingDagRunError::Invalid(
            "terminal publication has an invalid record shape".into(),
        ));
    };
    if artifact_stream.media_type != MeshingChunkMediaType::SolverMeshArtifact
        || evidence_stream.media_type != MeshingChunkMediaType::MeshingEvidence
    {
        return Err(ExactMeshingDagRunError::Invalid(
            "terminal publication has invalid media types".into(),
        ));
    }
    let artifact = SolverMeshArtifact::canonical_decode(artifact_bytes)?;
    let evidence = MeshingEvidence::canonical_decode(evidence_bytes)?;
    evidence.validate(&artifact)?;
    Ok(ExactMeshingDagRunResult {
        root,
        result_objects,
        artifact,
        evidence,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum ExactMeshingDagRunError {
    #[error("meshing DAG contract failed: {0}")]
    Bridge(#[from] crate::MeshingExecutionError),
    #[error("meshing DAG stage execution failed: {0}")]
    Stage(#[from] MeshingStageExecutionError),
    #[error("meshing DAG terminal contract failed: {0}")]
    Contract(#[from] runmat_meshing_core::MeshingContractError),
    #[error("meshing DAG state is invalid: {0}")]
    Invalid(String),
}

pub fn execute_exact_meshing_dag<E: ExactMeshingDagExecutor>(
    run: ExactMeshingDagRun<'_>,
    executor: &mut E,
) -> Result<ExactMeshingDagRunResult, ExactMeshingDagRunError> {
    futures::executor::block_on(execute_exact_meshing_dag_async(run, executor))
}

/// Executes the canonical exact-meshing graph without imposing a blocking backend contract.
///
/// This is the single orchestration authority used by both the synchronous entry point and
/// asynchronous local or remote executors.
pub async fn execute_exact_meshing_dag_async<E: ExactMeshingDagExecutor>(
    run: ExactMeshingDagRun<'_>,
    executor: &mut E,
) -> Result<ExactMeshingDagRunResult, ExactMeshingDagRunError> {
    let mut run = run;
    // Authored size controls become scoped metric contributions on persistent entities; every
    // stage then reads them from the one resolved request.
    run.request = run
        .request
        .resolve_size_controls(
            &run.named_regions,
            &run.geometry.geometry_objects().topology,
        )
        .map_err(|error| ExactMeshingDagRunError::Invalid(error.to_string()))?
        .0;
    validate_run(&run)?;
    register_inputs(&run, executor)?;
    let started = Instant::now();
    let planner = ExactMeshingDagPlanner::new(
        run.geometry,
        run.request.clone(),
        run.ceilings,
        run.artifact_access.clone(),
        run.capability_cohort.clone(),
    )?;
    let topology = &run.geometry.geometry_objects().topology;
    let curve_pass = planner.initial_curve_pass(topology, run.preferred_edges_per_partition)?;
    let mut observations = Vec::new();
    let curve_partitions = execute_partitions(
        executor,
        curve_pass.partitions(),
        run.program_revision.clone(),
        run.inventory_limits,
        &mut observations,
    )
    .await?;
    let curve_join = planner.curve_join(&curve_pass, curve_partitions)?;
    let curve = execute_planned_stage(
        executor,
        &curve_join,
        run.program_revision.clone(),
        run.inventory_limits,
    )
    .await?;
    observations.push(curve.evidence.clone());

    let surface = surface::execute_surface_dag(
        executor,
        &planner,
        topology,
        curve.root,
        &run,
        &mut observations,
    )
    .await?;
    terminal::execute_terminal_dag(executor, &planner, &run, surface, observations, started).await
}

async fn execute_partitions<E: ExactMeshingDagExecutor>(
    executor: &mut E,
    stages: &[crate::PlannedMeshingStage],
    revision: ProgramRevision,
    limits: ObjectInventoryLimits,
    observations: &mut Vec<runmat_meshing_core::MeshingStageEvidence>,
) -> Result<Vec<runmat_execution::value::ValueRef>, ExactMeshingDagRunError> {
    let successes = executor.execute_stages(stages, revision).await?;
    if successes.len() != stages.len() {
        return Err(crate::MeshingExecutionError::Invalid(
            "meshing executor returned the wrong partition result count".into(),
        )
        .into());
    }
    stages
        .iter()
        .zip(successes)
        .map(|(stage, success)| {
            let completed = admit_stage_success(executor, stage, success, limits)?;
            observations.push(completed.evidence);
            Ok(completed.root)
        })
        .collect()
}

fn validate_run(run: &ExactMeshingDagRun<'_>) -> Result<(), ExactMeshingDagRunError> {
    if run.preferred_edges_per_partition == 0 || run.preferred_faces_per_partition == 0 {
        return Err(ExactMeshingDagRunError::Invalid(
            "edge and face partition sizes must be nonzero".into(),
        ));
    }
    let topology = &run.geometry.geometry_objects().topology;
    if (run.request.dimension.is_solid() && topology.regions.is_empty())
        || (!run.request.dimension.is_solid() && topology.faces.is_empty())
    {
        return Err(ExactMeshingDagRunError::Stage(
            MeshingStageExecutionError::Failure(Box::new(MeshingFailure {
                schema_version: MESHING_FAILURE_SCHEMA_VERSION,
                category: MeshingFailureCategory::InvalidGeometry,
                stage: MeshingStageKind::GeometryAdmission,
                operation: MeshingOperation::AdmitGeometry,
                entity_ids: Vec::new(),
                witnesses: Vec::new(),
                request_values: Vec::new(),
                achieved_values: Vec::new(),
                accounting_error: None,
                remediation: if run.request.dimension.is_solid() {
                    "select geometry containing at least one closed solid region".into()
                } else {
                    "select geometry containing at least one face for shell or beam meshing".into()
                },
            })),
        ));
    }
    Ok(())
}

fn register_inputs<E: CacheExport>(
    run: &ExactMeshingDagRun<'_>,
    executor: &mut E,
) -> Result<(), ExactMeshingDagRunError> {
    for object in &run.geometry.geometry_objects().objects {
        executor
            .write_verified(object)
            .map_err(crate::MeshingExecutionError::from)?;
    }
    Ok(())
}
