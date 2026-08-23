//! One-call native composition of exact geometry admission and the canonical meshing DAG.

use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_io::{ExactCadImportOptions, GeometryImportContext};
use runmat_meshing_core::{MeshingDomainModel, MeshingRequest};
use runmat_meshing_execution::{
    execute_exact_meshing_dag, prepare_domain_model_input, prepare_domain_model_objects,
    ExactMeshingDagRun, ExactMeshingDagRunError, ExactMeshingDagRunResult, MeshingArtifactAccess,
    MeshingRunEvidenceContext,
};

use crate::{
    admit_exact_geometry, ExactGeometryAdmissionError, NativeExactMeshingExecutor,
    NativeMeshingExecutionPolicy, NativeProgramSession,
};

pub struct NativeExactMeshingJob<'a> {
    pub source_name: &'a str,
    pub source_bytes: &'a [u8],
    pub import_options: ExactCadImportOptions,
    pub request: MeshingRequest,
    pub domain_model: MeshingDomainModel,
    pub program_revision: ProgramRevision,
    pub capability_cohort: Option<String>,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
    pub inventory_limits: ObjectInventoryLimits,
    pub evidence: MeshingRunEvidenceContext,
    pub execution: NativeMeshingExecutionPolicy,
}

#[derive(Debug, thiserror::Error)]
pub enum NativeExactMeshingJobError {
    #[error(transparent)]
    Admission(#[from] ExactGeometryAdmissionError),
    #[error("meshing request tolerance differs from exact geometry import policy")]
    ToleranceMismatch,
    #[error("domain-model artifact preparation failed: {0}")]
    DomainModel(#[from] runmat_meshing_execution::MeshingExecutionError),
    #[error("domain-model object persistence failed: {0}")]
    Store(#[from] runmat_execution_artifact::ArtifactError),
    #[error("native meshing executor configuration failed: {0}")]
    Executor(#[from] runmat_meshing_execution::MeshingStageExecutionError),
    #[error(transparent)]
    Run(#[from] ExactMeshingDagRunError),
}

/// Executes exact CAD import through atomic solver-artifact publication in one native session.
///
/// Execution authority is derived from the session, so ordinary local use requires no Server or
/// user-managed artifact credentials. The returned artifact identity remains independent of that
/// physical authority.
pub fn mesh_exact_geometry(
    session: &NativeProgramSession,
    mut job: NativeExactMeshingJob<'_>,
) -> Result<ExactMeshingDagRunResult, NativeExactMeshingJobError> {
    let access = session_artifact_access(session);
    let geometry = admit_exact_geometry(
        session,
        job.source_name,
        job.source_bytes,
        &job.import_options,
        &GeometryImportContext::new(),
        access.clone(),
        job.inventory_limits,
    )?;
    let admitted_tolerance = geometry.geometry_objects().document.tolerance;
    if !same_requested_tolerance(&job.request.tolerance, &admitted_tolerance) {
        return Err(NativeExactMeshingJobError::ToleranceMismatch);
    }
    job.request.tolerance = admitted_tolerance;

    let domain_objects = prepare_domain_model_objects(job.domain_model, job.inventory_limits)?;
    let domain = prepare_domain_model_input(domain_objects, access.clone(), job.inventory_limits)?;
    let mut store = session.object_store();
    for object in &domain.domain_model_objects().objects {
        store.write_verified(object)?;
    }
    let mut executor = NativeExactMeshingExecutor::new(session, job.execution)?;
    execute_exact_meshing_dag(
        ExactMeshingDagRun {
            geometry: &geometry,
            domain_model: &domain,
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
    )
    .map_err(NativeExactMeshingJobError::from)
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
