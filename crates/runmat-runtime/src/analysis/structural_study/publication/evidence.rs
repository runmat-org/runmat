//! Immutable study-evidence persistence and content binding.
use std::path::PathBuf;

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
    let payload = payload
        .as_object()
        .cloned()
        .ok_or_else(|| "study evidence payload must be a JSON object".to_string())?;
    if payload.contains_key("$evidence") {
        return Err("study evidence payload may not define the reserved `$evidence` field".into());
    }

    let content_digest = payload_digest(&payload)?;
    let path_identity = if content_addressed {
        format!(
            "{stage}_sha256_{}",
            content_digest.trim_start_matches("sha256:")
        )
    } else {
        stage.to_string()
    };
    let evidence_root = study_evidence_root();
    storage::ensure_owned_directory(&evidence_root, "study evidence root")?;
    let root = evidence_root.join(format!("sha256_{digest}"));
    storage::ensure_owned_directory(&root, "study evidence fingerprint directory")?;
    let path = root.join(format!("{path_identity}.json"));
    let evidence = PersistedStudyEvidence {
        binding: StudyEvidenceBinding {
            schema_version: STUDY_EVIDENCE_SCHEMA_VERSION.to_string(),
            study_fingerprint: study_fingerprint.to_string(),
            stage: stage.to_string(),
            content_digest,
        },
        payload,
    };
    let bytes = serde_json::to_vec_pretty(&evidence)
        .map_err(|error| format!("failed to encode study evidence envelope: {error}"))?;
    match runmat_filesystem::atomic_create_new(&path, &bytes)
        .map_err(|error| format!("failed to publish immutable study evidence: {error}"))?
    {
        runmat_filesystem::AtomicCreateResult::Published { .. } => {}
        runmat_filesystem::AtomicCreateResult::Existing(existing_bytes) => {
            let existing = serde_json::from_slice::<PersistedStudyEvidence>(&existing_bytes)
                .map_err(|error| format!("failed to parse existing study evidence: {error}"))?;
            validate_evidence(&existing, study_fingerprint, stage)?;
            if existing != evidence {
                return Err(format!(
                    "study evidence identity already contains different content at {}",
                    path.display()
                ));
            }
        }
    }
    Ok(path.display().to_string())
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
