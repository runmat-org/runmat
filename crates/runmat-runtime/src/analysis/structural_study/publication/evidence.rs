//! Immutable study-evidence persistence and content binding.
use std::path::PathBuf;

mod access;
pub(in crate::analysis) use access::{immutable_batch_workspace, StudyEvidenceAccess};

use crate::analysis::{current_fea_runtime_config, default_fea_artifact_root, storage};

const STUDY_EVIDENCE_SCHEMA_VERSION: &str = "analysis_study_evidence/2026-09";

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StudyEvidenceBinding {
    schema_version: String,
    study_fingerprint: String,
    stage: String,
    content_digest: String,
}

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct PersistedStudyEvidence {
    #[serde(rename = "$evidence")]
    binding: StudyEvidenceBinding,
    // Study-result readers consume these fields directly. Keep that artifact shape while reserving
    // a typed namespace for the immutable identity and content binding.
    #[serde(flatten)]
    payload: serde_json::Map<String, serde_json::Value>,
}

pub(in crate::analysis) fn study_evidence_root() -> PathBuf {
    let config = current_fea_runtime_config();
    config
        .study_artifact_root
        .or_else(|| {
            std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT")
                .or_else(|_| std::env::var("RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT"))
                .ok()
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| {
            config
                .artifact_root
                .unwrap_or_else(default_fea_artifact_root)
                .join("studies")
        })
}

#[cfg(test)]
pub(in crate::analysis) fn persist_immutable_study_evidence(
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
) -> Result<String, String> {
    persist_study_evidence(study_fingerprint, stage, payload, false)
}

pub(in crate::analysis) fn persist_content_addressed_study_evidence(
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
) -> Result<String, String> {
    persist_study_evidence(study_fingerprint, stage, payload, true)
}

fn persist_study_evidence(
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
    content_addressed: bool,
) -> Result<String, String> {
    let maximum = crate::analysis::configuration::publication_staging_budget()?;
    persist_with_access(
        &StudyEvidenceAccess::capture(),
        study_fingerprint,
        stage,
        payload,
        content_addressed,
        maximum,
    )
    .map_err(|error| error.to_string())
}

pub(in crate::analysis) fn persist_admitted(
    access: &StudyEvidenceAccess,
    fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
    maximum: u64,
) -> Result<String, storage::AnalysisPublicationError> {
    persist_with_access(access, fingerprint, stage, payload, true, maximum)
}

fn persist_with_access(
    access: &StudyEvidenceAccess,
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
    content_addressed: bool,
    maximum: u64,
) -> Result<String, storage::AnalysisPublicationError> {
    let digest = study_fingerprint
        .strip_prefix("sha256:")
        .filter(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        })
        .ok_or_else(|| "study fingerprint must be a lowercase SHA-256 digest".to_string())?;
    if stage.is_empty()
        || stage.len() > 128
        || !stage.is_ascii()
        || !stage
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err("study evidence stage must be a safe 1-128 byte ASCII token".into());
    }
    let serde_json::Value::Object(payload) = payload else {
        return Err("study evidence payload must be a JSON object".into());
    };
    if payload.contains_key("$evidence") {
        return Err("study evidence payload may not define the reserved `$evidence` field".into());
    }

    let resident = storage::publication::resident(&payload, maximum)?;
    if resident > maximum {
        return Err(storage::AnalysisPublicationError::ResourceLimit {
            message: "study evidence input exceeds its phase allowance".into(),
        });
    }
    let payload_encoded = runmat_canonical_codec::profile_json_bytes(&payload, maximum)?;
    let encoding_allowance = maximum
        .checked_sub(resident)
        .filter(|bytes| *bytes >= payload_encoded)
        .ok_or_else(|| storage::AnalysisPublicationError::ResourceLimit {
            message: "evidence identity encoding exceeds its phase allowance".into(),
        })?;
    let payload_bytes = runmat_canonical_codec::encode_json_bounded(&payload, encoding_allowance)?;
    let content_digest = runmat_execution::Digest::sha256(&payload_bytes).to_string();
    drop(payload_bytes);
    let path_identity = if content_addressed {
        format!(
            "{stage}_sha256_{}",
            content_digest.trim_start_matches("sha256:")
        )
    } else {
        stage.to_string()
    };
    let evidence = PersistedStudyEvidence {
        binding: StudyEvidenceBinding {
            schema_version: STUDY_EVIDENCE_SCHEMA_VERSION.to_string(),
            study_fingerprint: study_fingerprint.to_string(),
            stage: stage.to_string(),
            content_digest,
        },
        payload,
    };
    let envelope_resident = storage::publication::resident(&evidence, maximum)?;
    access.persist(
        &format!("sha256_{digest}/{path_identity}.json"),
        &evidence,
        envelope_resident,
        maximum,
    )
}

fn validate_evidence(
    evidence: &PersistedStudyEvidence,
    expected_study_fingerprint: &str,
    expected_stage: &str,
) -> Result<(), String> {
    if evidence.binding.schema_version != STUDY_EVIDENCE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported study evidence schema `{}`",
            evidence.binding.schema_version
        ));
    }
    if evidence.binding.study_fingerprint != expected_study_fingerprint
        || evidence.binding.stage != expected_stage
    {
        return Err("study evidence path identity does not match its envelope".into());
    }
    if evidence.binding.content_digest != payload_digest(&evidence.payload)? {
        return Err("study evidence content digest does not match its payload".into());
    }
    Ok(())
}

fn payload_digest(payload: &serde_json::Map<String, serde_json::Value>) -> Result<String, String> {
    let payload_bytes = serde_json::to_vec(payload)
        .map_err(|error| format!("failed to encode study evidence payload: {error}"))?;
    Ok(runmat_execution::Digest::sha256(&payload_bytes).to_string())
}

pub(in crate::analysis) fn run_evidence_stage(run_id: &str) -> String {
    let digest = runmat_execution::Digest::sha256(run_id.as_bytes());
    format!("run-{}", digest.to_string().trim_start_matches("sha256:"))
}

/// Read only an explicitly named, content-addressed envelope inside the configured store.
/// Bound the stream itself as well as its metadata, so a changing file cannot bypass the limit.
#[derive(Debug, thiserror::Error)]
pub(in crate::analysis) enum StudyEvidenceReadError {
    #[error("study evidence exceeds its byte budget")]
    ResourceLimit,
    #[error("invalid study evidence: {0}")]
    Invalid(String),
}
impl From<String> for StudyEvidenceReadError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}
impl From<&str> for StudyEvidenceReadError {
    fn from(message: &str) -> Self {
        Self::Invalid(message.into())
    }
}
