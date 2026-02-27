use base64::Engine;
use chrono::Utc;
use runmat_builtins::Value;
use serde::{Deserialize, Serialize};

use crate::builtins::io::mat::load::decode_workspace_from_mat_bytes;
use crate::builtins::io::mat::save::encode_workspace_to_mat_bytes;
use crate::replay::limits::ReplayLimits;
use crate::runtime_error::{replay_error, replay_error_with_source, ReplayErrorKind};
use crate::{BuiltinResult, RuntimeError};

const WORKSPACE_SCHEMA_VERSION: u32 = 1;
const WORKSPACE_KIND: &str = "workspace-state";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceReplayMode {
    Auto,
    Force,
    Off,
}

impl WorkspaceReplayMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Force => "force",
            Self::Off => "off",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceReplayPayload {
    schema_version: u32,
    kind: String,
    created_at: String,
    mode: String,
    mat_base64: String,
}

pub async fn encode_workspace_payload(
    entries: &[(String, Value)],
    mode: &str,
) -> BuiltinResult<Vec<u8>> {
    encode_workspace_payload_with_limits(entries, mode, ReplayLimits::default()).await
}

pub async fn export_workspace_state(
    entries: &[(String, Value)],
    mode: WorkspaceReplayMode,
) -> BuiltinResult<Option<Vec<u8>>> {
    if matches!(mode, WorkspaceReplayMode::Off) {
        return Ok(None);
    }
    encode_workspace_payload(entries, mode.as_str())
        .await
        .map(Some)
}

pub async fn encode_workspace_payload_with_limits(
    entries: &[(String, Value)],
    mode: &str,
    limits: ReplayLimits,
) -> BuiltinResult<Vec<u8>> {
    validate_workspace_mode(mode)?;
    if entries.len() > limits.max_workspace_variables {
        return Err(replay_error(
            ReplayErrorKind::ImportRejected,
            format!(
                "workspace export includes {} variables, exceeding limit {}",
                entries.len(),
                limits.max_workspace_variables
            ),
        ));
    }

    let mat_bytes = encode_workspace_to_mat_bytes(entries).await?;
    if mat_bytes.len() > limits.max_workspace_mat_bytes {
        return Err(replay_error(
            ReplayErrorKind::PayloadTooLarge,
            format!(
                "workspace MAT payload is {} bytes, exceeding limit {}",
                mat_bytes.len(),
                limits.max_workspace_mat_bytes
            ),
        ));
    }

    let payload = WorkspaceReplayPayload {
        schema_version: WORKSPACE_SCHEMA_VERSION,
        kind: WORKSPACE_KIND.to_string(),
        created_at: Utc::now().to_rfc3339(),
        mode: mode.to_string(),
        mat_base64: base64::engine::general_purpose::STANDARD.encode(mat_bytes),
    };

    let encoded = serde_json::to_vec(&payload).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            "failed to encode workspace replay payload",
            err,
        )
    })?;

    if encoded.len() > limits.max_workspace_payload_bytes {
        return Err(replay_error(
            ReplayErrorKind::PayloadTooLarge,
            format!(
                "workspace replay payload is {} bytes, exceeding limit {}",
                encoded.len(),
                limits.max_workspace_payload_bytes
            ),
        ));
    }

    Ok(encoded)
}

pub fn decode_workspace_payload(bytes: &[u8]) -> BuiltinResult<Vec<(String, Value)>> {
    decode_workspace_payload_with_limits(bytes, ReplayLimits::default())
}

pub fn import_workspace_state(bytes: &[u8]) -> BuiltinResult<Vec<(String, Value)>> {
    decode_workspace_payload(bytes)
}

pub fn decode_workspace_payload_with_limits(
    bytes: &[u8],
    limits: ReplayLimits,
) -> BuiltinResult<Vec<(String, Value)>> {
    if bytes.len() > limits.max_workspace_payload_bytes {
        return Err(replay_error(
            ReplayErrorKind::PayloadTooLarge,
            format!(
                "workspace replay payload is {} bytes, exceeding limit {}",
                bytes.len(),
                limits.max_workspace_payload_bytes
            ),
        ));
    }

    let payload: WorkspaceReplayPayload = serde_json::from_slice(bytes).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            "failed to decode workspace replay payload",
            err,
        )
    })?;

    if payload.schema_version != WORKSPACE_SCHEMA_VERSION {
        return Err(replay_error(
            ReplayErrorKind::UnsupportedSchema,
            format!(
                "unsupported workspace replay schema version {}",
                payload.schema_version
            ),
        ));
    }
    if payload.kind != WORKSPACE_KIND {
        return Err(replay_error(
            ReplayErrorKind::ImportRejected,
            format!("unexpected replay payload kind '{}'", payload.kind),
        ));
    }
    validate_workspace_mode(&payload.mode)?;

    let mat_bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.mat_base64.as_bytes())
        .map_err(|err| {
            replay_error_with_source(
                ReplayErrorKind::DecodeFailed,
                "failed to decode workspace replay MAT bytes",
                err,
            )
        })?;

    if mat_bytes.len() > limits.max_workspace_mat_bytes {
        return Err(replay_error(
            ReplayErrorKind::PayloadTooLarge,
            format!(
                "workspace MAT payload is {} bytes, exceeding limit {}",
                mat_bytes.len(),
                limits.max_workspace_mat_bytes
            ),
        ));
    }

    let entries = decode_workspace_from_mat_bytes(&mat_bytes).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            "failed to decode workspace MAT payload",
            err,
        )
    })?;
    if entries.len() > limits.max_workspace_variables {
        return Err(replay_error(
            ReplayErrorKind::ImportRejected,
            format!(
                "workspace payload includes {} variables, exceeding limit {}",
                entries.len(),
                limits.max_workspace_variables
            ),
        ));
    }
    Ok(entries)
}

fn validate_workspace_mode(mode: &str) -> Result<(), RuntimeError> {
    if matches!(mode, "auto" | "force") {
        Ok(())
    } else {
        Err(replay_error(
            ReplayErrorKind::ImportRejected,
            format!("workspace replay mode '{mode}' is not supported"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;

    #[test]
    fn workspace_schema_mismatch_rejects() {
        let payload = serde_json::json!({
            "schemaVersion": 99,
            "kind": WORKSPACE_KIND,
            "createdAt": "2026-01-01T00:00:00Z",
            "mode": "auto",
            "matBase64": ""
        });
        let bytes = serde_json::to_vec(&payload).expect("serialize payload");
        let err = decode_workspace_payload_with_limits(&bytes, ReplayLimits::default())
            .expect_err("expected schema rejection");
        assert_eq!(
            err.identifier(),
            Some(ReplayErrorKind::UnsupportedSchema.identifier())
        );
    }

    #[test]
    fn workspace_payload_too_large_rejects() {
        let bytes = vec![0u8; ReplayLimits::default().max_workspace_payload_bytes + 1];
        let err = decode_workspace_payload_with_limits(&bytes, ReplayLimits::default())
            .expect_err("expected payload rejection");
        assert_eq!(
            err.identifier(),
            Some(ReplayErrorKind::PayloadTooLarge.identifier())
        );
    }

    #[test]
    fn workspace_variable_count_limit_rejects() {
        let entries = vec![
            ("a".to_string(), Value::Num(1.0)),
            ("b".to_string(), Value::Num(2.0)),
        ];
        let bytes = block_on(encode_workspace_payload_with_limits(
            &entries,
            "auto",
            ReplayLimits {
                max_workspace_variables: 4,
                ..ReplayLimits::default()
            },
        ))
        .expect("encode workspace payload");

        let err = decode_workspace_payload_with_limits(
            &bytes,
            ReplayLimits {
                max_workspace_variables: 1,
                ..ReplayLimits::default()
            },
        )
        .expect_err("expected variable limit rejection");
        assert_eq!(
            err.identifier(),
            Some(ReplayErrorKind::ImportRejected.identifier())
        );
    }
}
