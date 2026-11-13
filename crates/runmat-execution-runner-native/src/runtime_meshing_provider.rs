use std::path::{Path, PathBuf};

use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{
    resolve_meshing_request, CacheAdmissionDecision, CanonicalMeshingContract, MeshingRequest,
    MeshingRequestSettings,
};
use runmat_meshing_execution::{
    ExactMeshingDagRunError, MeshingRunEvidenceContext, MeshingStageExecutionError,
};
use runmat_runtime::meshing::{
    MeshingExecutionOutput, MeshingExecutionProvider, MeshingExecutionProviderError,
    MeshingExecutionRequest,
};

use crate::meshing_platform::{native_meshing_capability_cohort, native_meshing_platform};
use crate::{
    mesh_exact_geometry, MeshingExecutionPolicy, NativeExactMeshingJob, NativeExactMeshingJobError,
    NativeExactMeshingResult, NativeExecutionConfig, NativeMeshingContactDefinition,
};

pub struct NativeMeshingSourceRequest {
    pub source: PathBuf,
    pub import_options: ExactCadImportOptions,
    pub contact_definitions: Vec<NativeMeshingContactDefinition>,
    pub expected_source_sha256: Option<[u8; 32]>,
    pub settings: MeshingRequestSettings,
    pub partition_policy: NativeMeshingPartitionPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeMeshingPartitionPolicy {
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum NativeMeshingSourceError {
    #[error("{code}: {0}", code = .0.error_code())]
    RunAdmission(#[from] runmat_runtime::analysis::fea_run_admission::FeaRunDenied),
    #[error("failed to read {path}: {reason}")]
    SourceRead { path: PathBuf, reason: String },
    #[error("geometry source {path} changed after geometry.load; load the geometry again before meshing")]
    SourceDigestMismatch { path: PathBuf },
    #[error("invalid canonical meshing request: {0}")]
    InvalidRequest(String),
    #[error("failed to encode canonical meshing request: {0}")]
    RequestEncoding(String),
    #[error("failed to locate the RunMat meshing worker executable: {0}")]
    WorkerConfiguration(#[source] std::io::Error),
    #[error("exact meshing failed: {0}")]
    Meshing(#[from] NativeExactMeshingJobError),
}

impl NativeMeshingSourceError {
    pub const fn admission_error(&self) -> Option<&crate::ExactGeometryAdmissionError> {
        match self {
            Self::Meshing(error) => error.admission_error(),
            _ => None,
        }
    }

    pub const fn meshing_failure(&self) -> Option<&runmat_meshing_core::MeshingFailure> {
        match self {
            Self::Meshing(error) => error.meshing_failure(),
            _ => None,
        }
    }

    fn into_provider_error(self) -> MeshingExecutionProviderError {
        match self {
            Self::Meshing(NativeExactMeshingJobError::Executor(
                MeshingStageExecutionError::Failure(failure),
            ))
            | Self::Meshing(NativeExactMeshingJobError::Run(ExactMeshingDagRunError::Stage(
                MeshingStageExecutionError::Failure(failure),
            ))) => MeshingExecutionProviderError::Failure(failure),
            other => MeshingExecutionProviderError::Host(Box::new(other)),
        }
    }
}

impl Default for NativeMeshingPartitionPolicy {
    fn default() -> Self {
        Self {
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
        }
    }
}

pub fn native_meshing_execution_provider() -> MeshingExecutionProvider {
    std::sync::Arc::new(|request: &MeshingExecutionRequest| {
        mesh_source_with_native_execution(NativeMeshingSourceRequest {
            source: request.source_path.clone(),
            import_options: ExactCadImportOptions {
                source_units: request.source_units,
                ..ExactCadImportOptions::default()
            },
            contact_definitions: request
                .contact_definitions
                .iter()
                .map(|definition| NativeMeshingContactDefinition {
                    side_a_face_selectors: definition.side_a_face_selectors.clone(),
                    side_b_face_selectors: definition.side_b_face_selectors.clone(),
                })
                .collect(),
            expected_source_sha256: Some(request.expected_source_sha256),
            settings: request.settings.clone(),
            partition_policy: NativeMeshingPartitionPolicy::default(),
        })
        .map_err(NativeMeshingSourceError::into_provider_error)
        .map(|result| {
            let NativeExactMeshingResult { dag_result, .. } = result;
            MeshingExecutionOutput {
                artifact: dag_result.artifact,
                evidence: dag_result.evidence,
            }
        })
    })
}

pub fn mesh_source_with_native_execution(
    request: NativeMeshingSourceRequest,
) -> Result<NativeExactMeshingResult, NativeMeshingSourceError> {
    let source_bytes = runmat_filesystem::read(&request.source).map_err(|error| {
        NativeMeshingSourceError::SourceRead {
            path: request.source.clone(),
            reason: error.to_string(),
        }
    })?;
    verify_expected_source_sha256(
        &request.source,
        &source_bytes,
        request.expected_source_sha256,
    )?;
    let import_options = request.import_options;
    let canonical_request = resolve_native_meshing_request(&import_options, request.settings)?;
    let request_bytes = canonical_request
        .canonical_encode()
        .map_err(|error| NativeMeshingSourceError::RequestEncoding(error.to_string()))?;
    let cohort = native_meshing_capability_cohort();
    let program_revision = ProgramRevision::new(
        Digest::sha256(&request_bytes),
        Digest::sha256(&source_bytes),
        // The executing workers advertise this runtime's own environment.
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .map_err(|error| NativeMeshingSourceError::InvalidRequest(error.to_string()))?;
    let source_name = request.source.to_string_lossy().into_owned();
    use runmat_runtime::analysis::fea_run_admission::{
        FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope,
    };
    let mut execution = MeshingExecutionPolicy::default();
    if let Some(allocation) =
        runmat_runtime::analysis::fea_run_admission::current_fea_cpu_capacity_millicores()
    {
        execution.cpu_millicores = u32::try_from(allocation).map_err(|_| {
            NativeMeshingSourceError::InvalidRequest(
                "meshing CPU allocation exceeds its bound".into(),
            )
        })?;
    }
    let mut admission_request = FeaRunRequest::local(
        FeaRunKind::Meshing,
        u64::from(execution.cpu_millicores),
        Digest::sha256(&request_bytes),
    )?;
    admission_request.site = FeaRunScope::admission_site()?;
    let scope = FeaRunScope::admit(admission_request)?;
    let mut config = NativeExecutionConfig::for_current_executable()
        .map_err(NativeMeshingSourceError::WorkerConfiguration)?;
    // A per-stage budget is not an aggregate limit: independently ready stages
    // share the same admitted pool, rather than each spending the whole lease.
    config.local_cpu_capacity_millicores = Some(execution.cpu_millicores);
    config.worker_cpu_millicores = execution.cpu_millicores;
    config.max_workers = config
        .max_workers
        .min(execution.cpu_millicores / 1000)
        .max(1);
    scope.check()?;
    let result = scope.run_sync(|| {
        mesh_exact_geometry(
            config,
            NativeExactMeshingJob {
                source_name: &source_name,
                source_bytes: &source_bytes,
                import_options,
                contact_definitions: &request.contact_definitions,
                request: canonical_request,
                program_revision,
                capability_cohort: Some(cohort.clone()),
                preferred_edges_per_partition: request
                    .partition_policy
                    .preferred_edges_per_partition,
                preferred_faces_per_partition: request
                    .partition_policy
                    .preferred_faces_per_partition,
                inventory_limits: ObjectInventoryLimits::default(),
                evidence: MeshingRunEvidenceContext {
                    platform: native_meshing_platform(Some(&cohort), None),
                    sizing: Vec::new(),
                    cache_admission: CacheAdmissionDecision::Admitted,
                },
                execution,
            },
        )
        .map_err(NativeMeshingSourceError::from)
    });
    let outcome = if runmat_runtime::interrupt::is_cancelled() {
        FeaRunOutcome::Cancelled
    } else if result.is_ok() {
        FeaRunOutcome::Completed
    } else {
        FeaRunOutcome::Failed
    };
    let settlement = scope.settle(outcome);
    match result {
        Err(original) => Err(original),
        Ok(value) => settlement.map(|()| value).map_err(Into::into),
    }
}

pub fn resolve_native_meshing_request(
    import_options: &ExactCadImportOptions,
    settings: MeshingRequestSettings,
) -> Result<MeshingRequest, NativeMeshingSourceError> {
    let tolerance = runmat_geometry_core::GeometryTolerancePolicy {
        source_tolerance_m: runmat_types::physical::Length::new(0.0),
        absolute_floor_m: runmat_types::physical::Length::new(
            import_options
                .analysis
                .absolute_tolerance_floor_m
                .canonical_value(),
        ),
        model_relative_term: import_options.analysis.model_relative_tolerance,
        requested_deviation_m: runmat_types::physical::Length::new(
            import_options
                .analysis
                .requested_deviation_m
                .canonical_value(),
        ),
        maximum_healing_displacement_m: runmat_types::physical::Length::new(
            import_options
                .analysis
                .maximum_healing_displacement_m
                .canonical_value(),
        ),
    };
    resolve_meshing_request(tolerance, settings)
        .map_err(|error| NativeMeshingSourceError::InvalidRequest(error.to_string()))
}

fn verify_expected_source_sha256(
    path: &Path,
    bytes: &[u8],
    expected: Option<[u8; 32]>,
) -> Result<(), NativeMeshingSourceError> {
    if expected.is_some_and(|expected| *Digest::sha256(bytes).bytes() != expected) {
        return Err(NativeMeshingSourceError::SourceDigestMismatch {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitted_geometry_digest_rejects_changed_source_bytes() {
        let expected = *Digest::sha256(b"original").bytes();
        let error =
            verify_expected_source_sha256(Path::new("part.step"), b"changed", Some(expected))
                .expect_err("changed geometry must fail");
        assert!(error.to_string().contains("changed after geometry.load"));
    }
}
