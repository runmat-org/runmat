//! One-call native composition of exact geometry admission and the canonical meshing DAG.

use std::collections::BTreeMap;

use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_io::{ExactCadImportOptions, GeometryImportContext};
use runmat_meshing_core::MeshingRequest;
use runmat_meshing_execution::{
    execute_exact_meshing_dag, ExactMeshingDagRun, ExactMeshingDagRunError,
    ExactMeshingDagRunResult, MeshingArtifactAccess, MeshingRunEvidenceContext,
};

use crate::{
    admit_prepared_exact_geometry, prepare_exact_geometry_admission, ExactGeometryAdmissionError,
    NativeExactMeshingExecutor, NativeExecutionConfig, NativeExecutionError,
    NativeMeshingExecutionPolicy, NativeProgramSession,
};

pub struct NativeExactMeshingJob<'a> {
    pub source_name: &'a str,
    pub source_bytes: &'a [u8],
    pub import_options: ExactCadImportOptions,
    pub request: MeshingRequest,
    pub program_revision: ProgramRevision,
    pub capability_cohort: Option<String>,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
    pub inventory_limits: ObjectInventoryLimits,
    pub evidence: MeshingRunEvidenceContext,
    pub execution: NativeMeshingExecutionPolicy,
}

pub struct NativeExactMeshingResult {
    pub dag_result: ExactMeshingDagRunResult,
    pub source_face_ids: BTreeMap<u64, runmat_geometry_core::PersistentEntityId>,
}

#[derive(Debug, thiserror::Error)]
pub enum NativeExactMeshingJobError {
    #[error(transparent)]
    Admission(#[from] ExactGeometryAdmissionError),
    #[error("meshing request tolerance differs from exact geometry import policy")]
    ToleranceMismatch,
    #[error("native meshing executor configuration failed: {0}")]
    Executor(#[from] runmat_meshing_execution::MeshingStageExecutionError),
    #[error("native execution session failed: {0}")]
    Session(#[from] NativeExecutionError),
    #[error(transparent)]
    Run(#[from] ExactMeshingDagRunError),
}

/// Executes exact CAD import through atomic solver-artifact publication in one native session.
///
/// Execution authority is derived from the session, so ordinary local use requires no Server or
/// user-managed artifact credentials. The returned artifact identity remains independent of that
/// physical authority.
pub fn mesh_exact_geometry(
    mut config: NativeExecutionConfig,
    mut job: NativeExactMeshingJob<'_>,
) -> Result<NativeExactMeshingResult, NativeExactMeshingJobError> {
    let prepared_geometry = prepare_exact_geometry_admission(
        job.source_name,
        job.source_bytes,
        &job.import_options,
        &GeometryImportContext::new(),
        job.inventory_limits,
    )?;
    let admitted_tolerance = prepared_geometry.document().tolerance;
    if !same_requested_tolerance(&job.request.tolerance, &admitted_tolerance) {
        return Err(NativeExactMeshingJobError::ToleranceMismatch);
    }
    job.request.tolerance = admitted_tolerance;
    job.evidence.platform.exact_kernel_abi =
        prepared_geometry.document().source.kernel_version.clone();
    let source_face_ids = prepared_geometry.source_face_ids().clone();
    config.enable_exact_meshing(
        prepared_geometry.document(),
        &job.request,
        job.capability_cohort.as_deref(),
    )?;
    let session = NativeProgramSession::new(config)?;
    let access = session_artifact_access(&session);
    let geometry = admit_prepared_exact_geometry(
        &session,
        prepared_geometry,
        access.clone(),
        job.inventory_limits,
    )?;

    let mut executor = NativeExactMeshingExecutor::new(&session, job.execution)?;
    let dag_result = execute_exact_meshing_dag(
        ExactMeshingDagRun {
            geometry: &geometry,
            request: job.request,
            artifact_access: access,
            capability_cohort: job.capability_cohort,
            program_revision: job.program_revision,
            preferred_edges_per_partition: job.preferred_edges_per_partition,
            preferred_faces_per_partition: job.preferred_faces_per_partition,
            inventory_limits: job.inventory_limits,
            evidence: job.evidence,
        },
        &mut executor,
    )?;
    Ok(NativeExactMeshingResult {
        dag_result,
        source_face_ids,
    })
}

fn session_artifact_access(session: &NativeProgramSession) -> MeshingArtifactAccess {
    MeshingArtifactAccess {
        authorization_scope: format!("native-meshing-session:{}", session.scope_id()),
        encryption_context: Digest::sha256(
            [
                b"runmat-native-meshing-session\0".as_slice(),
                session.scope_id().bytes(),
            ]
            .concat(),
        ),
    }
}

fn same_requested_tolerance(
    requested: &runmat_geometry_core::GeometryTolerancePolicy,
    admitted: &runmat_geometry_core::GeometryTolerancePolicy,
) -> bool {
    requested.absolute_floor_m == admitted.absolute_floor_m
        && requested.model_relative_term == admitted.model_relative_term
        && requested.requested_deviation_m == admitted.requested_deviation_m
        && requested.maximum_healing_displacement_m == admitted.maximum_healing_displacement_m
}

#[cfg(test)]
mod tests {
    use runmat_geometry_core::GeometryTolerancePolicy;

    use super::same_requested_tolerance;

    #[test]
    fn source_tolerance_is_resolved_by_import_but_user_policy_must_match() {
        let requested = GeometryTolerancePolicy {
            source_tolerance_m: 0.0,
            absolute_floor_m: 1.0e-12,
            model_relative_term: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 1.0e-6,
        };
        let mut admitted = requested;
        admitted.source_tolerance_m = 2.0e-8;
        assert!(same_requested_tolerance(&requested, &admitted));
        admitted.requested_deviation_m = 2.0e-4;
        assert!(!same_requested_tolerance(&requested, &admitted));
    }
}
