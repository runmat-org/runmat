//! Runtime composition boundary for canonical study meshing.

use std::path::{Path, PathBuf};

use runmat_meshing_core::CanonicalMeshingContract;
use sha2::{Digest as _, Sha256};

#[path = "meshing/persistence.rs"]
mod persistence;

use super::{
    atomic_write_bytes, current_fea_runtime_config, default_fea_artifact_root,
    solver_mesh_artifact, AnalysisStudySpec,
};

pub(crate) struct ResolvedStudyMesh {
    pub artifact_path: Option<String>,
    pub evidence_path: Option<String>,
}

pub(super) enum StudyMeshingError {
    Message(String),
    Execution(crate::meshing::MeshingExecutionError),
}

pub(super) fn resolve_study_mesh(
    spec: &AnalysisStudySpec,
) -> Result<ResolvedStudyMesh, StudyMeshingError> {
    if let Some(path) = spec.solver_mesh_artifact_path.as_deref() {
        solver_mesh_artifact::load_solver_mesh_artifact(Path::new(path))
            .map_err(|error| StudyMeshingError::Message(error.to_string()))?;
        return Ok(ResolvedStudyMesh {
            artifact_path: Some(path.to_owned()),
            evidence_path: spec.meshing_evidence_artifact_path.clone(),
        });
    }
    let Some(request) = study_meshing_request(spec).map_err(StudyMeshingError::Message)? else {
        return Ok(ResolvedStudyMesh {
            artifact_path: None,
            evidence_path: None,
        });
    };
    let output = crate::meshing::execute_meshing(&request).map_err(StudyMeshingError::Execution)?;
    validate_and_persist_mesh(&output).map_err(StudyMeshingError::Message)
}

/// Resolve the same original request through the host's owned async mesher.
pub(super) async fn resolve_study_mesh_async(
    spec: &AnalysisStudySpec,
    context: &crate::operations::OperationContext,
) -> Result<ResolvedStudyMesh, crate::operations::OperationErrorEnvelope> {
    let rejected = |message| {
        super::operation_error(
            super::ANALYSIS_RUN_STUDY_OPERATION,
            super::ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            super::OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.MESHING_FAILED",
                error_type: super::OperationErrorType::Backend,
                retryable: false,
                severity: super::OperationErrorSeverity::Error,
            },
            message,
            std::collections::BTreeMap::new(),
        )
    };
    if let Some(path) = &spec.solver_mesh_artifact_path {
        // The caller reads and canonically validates this original input through
        // its async provider before topology construction and numerical work.
        return Ok(ResolvedStudyMesh {
            artifact_path: Some(path.clone()),
            evidence_path: spec.meshing_evidence_artifact_path.clone(),
        });
    }
    let Some(request) = study_meshing_request(spec).map_err(rejected)? else {
        return Ok(ResolvedStudyMesh {
            artifact_path: None,
            evidence_path: None,
        });
    };
    let output = crate::meshing::execute_meshing_async(&request)
        .await
        .map_err(|error| async_meshing_error(error, context))?;
    super::storage::with_artifact_access(true, |_| {
        super::fea_run_admission::FeaRunScope::check_publication_for_operation(
            super::ANALYSIS_RUN_STUDY_OPERATION,
            super::ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
        )?;
        output
            .evidence
            .validate(&output.artifact)
            .map_err(|error| {
                rejected(format!(
                    "meshing provider returned invalid evidence: {error}"
                ))
            })?;
        persist_mesh_borrowed_typed(&output.artifact, &output.evidence)
            .map_err(|error| super::study_step_execution::study_artifact_error(error, context))
    })
    .await
    .map_err(|error| super::study_step_execution::study_artifact_error(error, context))?
}

pub(super) fn async_meshing_error(
    error: crate::meshing::MeshingExecutionError,
    context: &crate::operations::OperationContext,
) -> crate::operations::OperationErrorEnvelope {
    use crate::meshing::MeshingExecutionError;
    match error {
        MeshingExecutionError::Admission(denied) => denied.into_operation_error(
            super::ANALYSIS_RUN_STUDY_OPERATION,
            super::ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
        ),
        MeshingExecutionError::WithLeaseFailure { primary, lease } => {
            super::fea_run_admission::entrypoints::merge_lease_error::<()>(
                Err(async_meshing_error(*primary, context)),
                Some(lease.into_operation_error(
                    super::ANALYSIS_RUN_STUDY_OPERATION,
                    super::ANALYSIS_RUN_STUDY_OP_VERSION,
                    context,
                )),
            )
            .expect_err("the original meshing operation failed")
        }
        MeshingExecutionError::Provider(
            crate::meshing::MeshingExecutionProviderError::Failure(failure),
        ) if failure.accounting_error.is_some() => {
            let cause = failure
                .accounting_error
                .expect("accounting cause checked above");
            super::fea_run_admission::FeaRunDenied::Accounting(cause).into_operation_error(
                super::ANALYSIS_RUN_STUDY_OPERATION,
                super::ANALYSIS_RUN_STUDY_OP_VERSION,
                context,
            )
        }
        MeshingExecutionError::Provider(error) => super::operation_error(
            super::ANALYSIS_RUN_STUDY_OPERATION,
            super::ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            super::OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.MESHING_FAILED",
                error_type: if crate::interrupt::is_cancelled() {
                    super::OperationErrorType::Cancelled
                } else {
                    super::OperationErrorType::Backend
                },
                retryable: false,
                severity: super::OperationErrorSeverity::Error,
            },
            error.to_string(),
            std::collections::BTreeMap::new(),
        ),
    }
}

fn study_meshing_request(
    spec: &AnalysisStudySpec,
) -> Result<Option<crate::meshing::MeshingExecutionRequest>, String> {
    let Some(settings) = spec.meshing_settings.clone() else {
        return Ok(None);
    };
    Ok(Some(crate::meshing::MeshingExecutionRequest {
        source_path: PathBuf::from(&spec.geometry.source.path),
        source_units: match &spec.geometry.source.unit_evidence {
            runmat_geometry_core::physical::LengthUnitEvidence::ExplicitFallback {
                declaration,
                ..
            } => Some(declaration.unit),
            _ => None,
        },
        expected_source_sha256: crate::meshing::parse_source_sha256(&spec.geometry.source.sha256)?,
        settings,
        contact_definitions: meshing_contact_definitions(spec),
    }))
}

fn validate_and_persist_mesh(
    output: &crate::meshing::MeshingExecutionOutput,
) -> Result<ResolvedStudyMesh, String> {
    output
        .evidence
        .validate(&output.artifact)
        .map_err(|error| format!("meshing provider returned invalid evidence: {error}"))?;
    persist_mesh_output(output)
}

pub(super) fn meshing_contact_definitions(
    spec: &AnalysisStudySpec,
) -> Vec<crate::meshing::MeshingContactDefinition> {
    meshing_contact_definitions_for_model(&spec.model).collect()
}

pub(super) fn meshing_contact_definitions_for_model(
    model: &runmat_analysis_core::AnalysisModel,
) -> impl Iterator<Item = crate::meshing::MeshingContactDefinition> + '_ {
    model
        .interfaces
        .iter()
        .filter(|interface| {
            matches!(
                interface.kind,
                runmat_analysis_core::AnalysisInterfaceKind::Contact(_)
            )
        })
        .map(|interface| crate::meshing::MeshingContactDefinition {
            side_a_face_selectors: vec![interface.primary.region_id.clone()],
            side_b_face_selectors: vec![interface.secondary.region_id.clone()],
        })
}

pub(crate) fn persist_mesh_output(
    output: &crate::meshing::MeshingExecutionOutput,
) -> Result<ResolvedStudyMesh, String> {
    persist_mesh_borrowed(&output.artifact, &output.evidence)
}

/// Persist original admitted inventories without constructing a second mesh owner.
pub(crate) fn persist_mesh_borrowed(
    artifact: &runmat_meshing_core::SolverMeshArtifact,
    evidence: &runmat_meshing_core::MeshingEvidence,
) -> Result<ResolvedStudyMesh, String> {
    persist_mesh_borrowed_typed(artifact, evidence).map_err(|error| error.to_string())
}

fn persist_mesh_borrowed_typed(
    artifact: &runmat_meshing_core::SolverMeshArtifact,
    evidence: &runmat_meshing_core::MeshingEvidence,
) -> Result<ResolvedStudyMesh, super::storage::AnalysisPublicationError> {
    let artifact_digest = encode_digest(artifact.canonical_digest.bytes());
    let root = current_fea_runtime_config()
        .artifact_root
        .or_else(|| {
            std::env::var("RUNMAT_FEA_ARTIFACT_ROOT")
                .ok()
                .map(PathBuf::from)
        })
        .unwrap_or_else(default_fea_artifact_root)
        .join("meshes");
    let access = persistence::MeshPersistence::capture_typed(&root)?;
    let artifact_bytes = artifact
        .canonical_encode_with_cancellation(&MeshPersistenceCancellation)
        .map_err(|error| error.to_string())?;
    let evidence_bytes = evidence
        .canonical_encode()
        .map_err(|error| error.to_string())?;
    let evidence_digest = encode_digest(&Sha256::digest(&evidence_bytes).into());
    let artifact_path = root.join(format!("{artifact_digest}.solver-mesh.cbor"));
    let evidence_path = root.join(format!(
        "{artifact_digest}-{evidence_digest}.meshing-evidence.cbor"
    ));
    access.persist_typed(&artifact_path, &artifact_bytes)?;
    access.persist_typed(&evidence_path, &evidence_bytes)?;
    Ok(ResolvedStudyMesh {
        artifact_path: Some(artifact_path.display().to_string()),
        evidence_path: Some(evidence_path.display().to_string()),
    })
}

struct MeshPersistenceCancellation;
impl runmat_meshing_core::MeshingCancellationSignal for MeshPersistenceCancellation {
    fn is_cancelled(&self) -> bool {
        crate::interrupt::is_cancelled()
    }
}

fn persist_canonical(path: &Path, bytes: &[u8]) -> Result<(), String> {
    // The browser's artifacts belong to the installed provider, not std::fs.
    // Only an authoritative NotFound permits creating a new immutable artifact.
    let exists = match runmat_filesystem::metadata(path) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(format!("failed to inspect {}: {error}", path.display())),
    };
    if exists {
        let mut existing = runmat_filesystem::File::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        return compare_canonical(path, bytes, &mut existing);
    }
    atomic_write_bytes(&path.to_path_buf(), bytes)
}

fn compare_canonical(
    path: &Path,
    bytes: &[u8],
    existing: &mut impl std::io::Read,
) -> Result<(), String> {
    let mut offset = 0;
    let mut buffer = [0_u8; 8192];
    while offset < bytes.len() {
        if crate::interrupt::is_cancelled() {
            return Err("meshing persistence cancelled".into());
        }
        let count = buffer.len().min(bytes.len() - offset);
        let read = std::io::Read::read(existing, &mut buffer[..count])
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read == 0 || buffer[..read] != bytes[offset..offset + read] {
            break;
        }
        offset += read;
    }
    if offset == bytes.len()
        && std::io::Read::read(existing, &mut buffer[..1])
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?
            == 0
    {
        return Ok(());
    }
    Err(format!(
        "canonical meshing output collision at {}",
        path.display()
    ))
}

fn encode_digest(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}

#[cfg(test)]
#[path = "meshing/persistence_tests.rs"]
mod persistence_tests;
