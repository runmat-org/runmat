use std::path::{Path, PathBuf};

use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{
    resolve_meshing_request, CacheAdmissionDecision, CanonicalMeshingContract, MeshingRequest,
    MeshingRequestSettings,
};
use runmat_meshing_execution::MeshingRunEvidenceContext;
use runmat_runtime::meshing::{
    MeshingExecutionOutput, MeshingExecutionProvider, MeshingExecutionRequest,
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
        .map(|result| MeshingExecutionOutput {
            artifact: result.dag_result.artifact,
            evidence: result.dag_result.evidence,
            boundary_region_ids: result
                .source_face_ids
                .into_iter()
                .map(|(index, id)| (format!("face_{:06}", index + 1), id))
                .collect(),
        })
        .map_err(|error| error.to_string())
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
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(env!("CARGO_PKG_VERSION").as_bytes()),
            Digest::sha256(b"runmat-meshing-catalog/v1"),
            "matlab",
        )
        .map_err(|error| NativeMeshingSourceError::InvalidRequest(error.to_string()))?,
    )
    .map_err(|error| NativeMeshingSourceError::InvalidRequest(error.to_string()))?;
    let source_name = request.source.to_string_lossy().into_owned();
    mesh_exact_geometry(
        NativeExecutionConfig::for_current_executable()
            .map_err(NativeMeshingSourceError::WorkerConfiguration)?,
        NativeExactMeshingJob {
            source_name: &source_name,
            source_bytes: &source_bytes,
            import_options,
            contact_definitions: &request.contact_definitions,
            request: canonical_request,
            program_revision,
            capability_cohort: Some(cohort.clone()),
            preferred_edges_per_partition: request.partition_policy.preferred_edges_per_partition,
            preferred_faces_per_partition: request.partition_policy.preferred_faces_per_partition,
            inventory_limits: ObjectInventoryLimits::default(),
            evidence: MeshingRunEvidenceContext {
                platform: native_meshing_platform(Some(&cohort), None),
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
            execution: MeshingExecutionPolicy::default(),
        },
    )
    .map_err(NativeMeshingSourceError::from)
}

pub fn resolve_native_meshing_request(
    import_options: &ExactCadImportOptions,
    settings: MeshingRequestSettings,
) -> Result<MeshingRequest, NativeMeshingSourceError> {
    let tolerance = runmat_geometry_core::GeometryTolerancePolicy {
        source_tolerance_m: 0.0,
        absolute_floor_m: import_options.analysis.absolute_tolerance_floor_m,
        model_relative_term: import_options.analysis.model_relative_tolerance,
        requested_deviation_m: import_options.analysis.requested_deviation_m,
        maximum_healing_displacement_m: import_options.analysis.maximum_healing_displacement_m,
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
