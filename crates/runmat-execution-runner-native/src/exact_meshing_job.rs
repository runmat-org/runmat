//! One-call native composition of exact geometry admission and the canonical meshing DAG.

use std::collections::BTreeMap;

use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_geometry_io::{ExactCadImportOptions, GeometryImportContext};
use runmat_meshing_core::MeshingRequest;
use runmat_meshing_execution::{
    execute_exact_meshing_dag, ExactMeshingDagRun, ExactMeshingDagRunError,
    ExactMeshingDagRunResult, MeshingRunEvidenceContext,
};

use crate::exact_geometry_admission::same_requested_tolerance;
use crate::{
    admit_prepared_exact_geometry, prepare_exact_geometry_admission, ExactGeometryAdmissionError,
    MeshingExecutionPolicy, NativeExactMeshingExecutor, NativeExecutionConfig,
    NativeExecutionError, NativeMeshingContactDefinition, NativeProgramSession,
};

pub struct NativeExactMeshingJob<'a> {
    pub source_name: &'a str,
    pub source_bytes: &'a [u8],
    pub import_options: ExactCadImportOptions,
    pub contact_definitions: &'a [NativeMeshingContactDefinition],
    pub request: MeshingRequest,
    pub program_revision: ProgramRevision,
    pub capability_cohort: Option<String>,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
    pub inventory_limits: ObjectInventoryLimits,
    pub evidence: MeshingRunEvidenceContext,
    pub execution: MeshingExecutionPolicy,
}

pub struct NativeExactMeshingResult {
    pub dag_result: ExactMeshingDagRunResult,
    pub source_edge_ids: BTreeMap<u64, runmat_geometry_core::PersistentEntityId>,
    pub source_face_ids: BTreeMap<u64, runmat_geometry_core::PersistentEntityId>,
    pub source_region_ids: BTreeMap<u64, runmat_geometry_core::PersistentEntityId>,
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
    #[error("meshed volume domains do not preserve admitted CAD-region identity")]
    RegionIdentityMismatch,
}

impl NativeExactMeshingJobError {
    pub const fn admission_error(&self) -> Option<&ExactGeometryAdmissionError> {
        match self {
            Self::Admission(error) => Some(error),
            _ => None,
        }
    }

    pub const fn meshing_failure(&self) -> Option<&runmat_meshing_core::MeshingFailure> {
        match self {
            Self::Executor(error) => error.failure(),
            Self::Run(ExactMeshingDagRunError::Stage(error)) => error.failure(),
            _ => None,
        }
    }
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
    let prepared_geometry = {
        let span = tracing::info_span!(
            "meshing.geometry_admission",
            source_name = job.source_name,
            source_bytes = job.source_bytes.len(),
        );
        let _entered = span.enter();
        prepare_exact_geometry_admission(
            job.source_name,
            job.source_bytes,
            &job.import_options,
            job.contact_definitions,
            &GeometryImportContext::new(),
            job.inventory_limits,
        )?
    };
    let admitted_tolerance = prepared_geometry.document().tolerance;
    if !same_requested_tolerance(&job.request.tolerance, &admitted_tolerance) {
        return Err(NativeExactMeshingJobError::ToleranceMismatch);
    }
    job.request.tolerance = admitted_tolerance;
    job.evidence.platform.exact_kernel_abi =
        prepared_geometry.document().source.kernel_version.clone();
    let source_edge_ids = prepared_geometry.source_edge_ids().clone();
    let source_face_ids = prepared_geometry.source_face_ids().clone();
    let source_region_ids = prepared_geometry.source_region_ids().clone();
    // The run's execution ceilings come from this host's capacity; the local workers must be
    // able to admit tasks of that size.
    let ceilings = match config.local_memory_capacity_bytes {
        Some(capacity) => {
            crate::machine_capacity::meshing_execution_ceilings_for(
                capacity.min(config.authored_worker_memory_bytes.unwrap_or(u64::MAX)),
                &job.request.limits,
            )
        }
        None => crate::machine_capacity::meshing_execution_ceilings(&job.request.limits),
    };
    config.worker_memory_bytes = config
        .worker_memory_bytes
        .max(ceilings.maximum_memory_bytes);
    config.worker_scratch_bytes = config
        .worker_scratch_bytes
        .max(ceilings.maximum_scratch_bytes);
    let (session, access, geometry) = {
        let span = tracing::info_span!("meshing.native_session");
        let _entered = span.enter();
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
        (session, access, geometry)
    };

    let mut executor = NativeExactMeshingExecutor::new(
        &session,
        MeshingExecutionPolicy {
            reuse_accepted_stages: job.execution.reuse_accepted_stages
                && job.evidence.cache_admission
                    == runmat_meshing_core::CacheAdmissionDecision::Admitted,
            ..job.execution
        },
    )?;
    let dag_result = {
        let span = tracing::info_span!("meshing.exact_dag");
        let _entered = span.enter();
        execute_exact_meshing_dag(
            ExactMeshingDagRun {
                geometry: &geometry,
                ceilings,
                request: job.request,
                artifact_access: access,
                capability_cohort: job.capability_cohort,
                program_revision: job.program_revision,
                preferred_edges_per_partition: job.preferred_edges_per_partition,
                preferred_faces_per_partition: job.preferred_faces_per_partition,
                inventory_limits: job.inventory_limits,
                evidence: job.evidence,
                named_regions: Default::default(),
            },
            &mut executor,
        )?
    };
    let meshed_region_ids = dag_result
        .artifact
        .topology
        .regions
        .iter()
        .map(|region| region.region_id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let admitted_region_ids = source_region_ids
        .values()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if source_region_ids.len() != meshed_region_ids.len()
        || admitted_region_ids != meshed_region_ids
    {
        return Err(NativeExactMeshingJobError::RegionIdentityMismatch);
    }
    Ok(NativeExactMeshingResult {
        dag_result,
        source_edge_ids,
        source_face_ids,
        source_region_ids,
    })
}

fn session_artifact_access(session: &NativeProgramSession) -> ArtifactAccess {
    ArtifactAccess {
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

#[cfg(test)]
mod tests {
    use runmat_geometry_core::GeometryTolerancePolicy;

    use crate::exact_geometry_admission::same_requested_tolerance;

    #[test]
    fn source_tolerance_is_resolved_by_import_but_user_policy_must_match() {
        let requested = GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(0.0),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-12),
            model_relative_term: 1.0e-12,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-4),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        };
        let mut admitted = requested;
        admitted.source_tolerance_m = runmat_types::physical::Length::new(2.0e-8);
        assert!(same_requested_tolerance(&requested, &admitted));
        admitted.requested_deviation_m = runmat_types::physical::Length::new(2.0e-4);
        assert!(!same_requested_tolerance(&requested, &admitted));
    }
}
