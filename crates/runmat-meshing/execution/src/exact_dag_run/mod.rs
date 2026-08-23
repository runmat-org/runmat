//! Backend-neutral orchestration of the meshing-owned exact geometry DAG.

mod stage;
mod surface;
mod terminal;

use std::time::Instant;

use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_meshing_core::{
    CacheAdmissionDecision, MeshingEvidence, MeshingRequest, PlatformBuildIdentity,
    SizingResolutionEvidence, SolverMeshArtifact,
};

use crate::{
    ExactMeshingDagPlanner, MeshingArtifactAccess, PreparedDomainModelInput,
    PreparedExactGeometryInput,
};

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
    pub domain_model: &'a PreparedDomainModelInput,
    pub request: MeshingRequest,
    pub artifact_access: MeshingArtifactAccess,
    pub capability_cohort: Option<String>,
    pub program_revision: ProgramRevision,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
    pub inventory_limits: ObjectInventoryLimits,
    pub evidence: MeshingRunEvidenceContext,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactMeshingDagRunResult {
    pub root: runmat_execution::value::ValueRef,
    pub result_objects: Vec<runmat_execution::value::ValueRef>,
    pub artifact: SolverMeshArtifact,
    pub evidence: MeshingEvidence,
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
    validate_run(&run)?;
    register_inputs(&run, executor)?;
    let started = Instant::now();
    let planner = ExactMeshingDagPlanner::new(
        run.geometry,
        run.request.clone(),
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
    )?;
    let curve_join = planner.curve_join(&curve_pass, curve_partitions)?;
    let curve = execute_planned_stage(
        executor,
        &curve_join,
        run.program_revision.clone(),
        run.inventory_limits,
    )?;
    observations.push(curve.evidence.clone());

    let surface = surface::execute_surface_dag(
        executor,
        &planner,
        topology,
        curve.root,
        &run,
        &mut observations,
    )?;
    terminal::execute_terminal_dag(executor, &planner, &run, surface, observations, started)
}

fn execute_partitions<E: ExactMeshingDagExecutor>(
    executor: &mut E,
    stages: &[crate::PlannedMeshingStage],
    revision: ProgramRevision,
    limits: ObjectInventoryLimits,
    observations: &mut Vec<runmat_meshing_core::MeshingStageEvidence>,
) -> Result<Vec<runmat_execution::value::ValueRef>, ExactMeshingDagRunError> {
    let successes = executor.execute_stages(stages, revision)?;
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
    if run.domain_model.root_input().authorization_scope != run.artifact_access.authorization_scope
        || run.domain_model.root_input().encryption_context
            != run.artifact_access.encryption_context
        || run.domain_model.root_input().id
            != run
                .artifact_access
                .value_id(run.domain_model.root_input().logical_digest)
    {
        return Err(crate::MeshingExecutionError::Identity(
            "domain model root is outside the meshing artifact authority",
        )
        .into());
    }
    Ok(())
}

fn register_inputs<E: CacheExport>(
    run: &ExactMeshingDagRun<'_>,
    executor: &mut E,
) -> Result<(), ExactMeshingDagRunError> {
    for object in run
        .geometry
        .geometry_objects()
        .objects
        .iter()
        .chain(&run.domain_model.domain_model_objects().objects)
    {
        executor
            .write_verified(object)
            .map_err(crate::MeshingExecutionError::from)?;
    }
    Ok(())
}
