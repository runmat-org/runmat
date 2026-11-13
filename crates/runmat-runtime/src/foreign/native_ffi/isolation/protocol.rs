use runmat_execution::value::ValuePayload;
use runmat_execution::Digest;
use runmat_process_host::shared_memory::SharedMemoryDescriptor;
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use serde::{Deserialize, Serialize};

mod cause;
use cause::NativeWireCause;

pub const NATIVE_FFI_HOST_KIND: &str = "native-ffi";
pub const NATIVE_FFI_HOST_KIND_ENV: &str = "RUNMAT_EXTENSION_HOST_KIND";
pub const NATIVE_FFI_HOST_FRONTEND_ENV: &str = "RUNMAT_NATIVE_FFI_HOST_FRONTEND";
pub const NATIVE_FFI_HOST_PROTOCOL: &str = "runmat.native-ffi-host";
pub const NATIVE_FFI_HOST_SECRET_ENV: &str = "RUNMAT_NATIVE_FFI_HOST_SECRET";
pub const NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV: &str = "RUNMAT_NATIVE_FFI_SNAPSHOT_ROOT";
#[cfg(test)]
pub const NATIVE_FFI_HOST_SCHEMA_V1: u16 = 1;
pub const NATIVE_FFI_HOST_SCHEMA_V2: u16 = 2;
pub const NATIVE_FFI_HOST_SCHEMA_V3: u16 = 3;
pub const NATIVE_FFI_HOST_SCHEMA_VERSION: u16 = NATIVE_FFI_HOST_SCHEMA_V3;
pub const NATIVE_FFI_HOST_MAX_MESSAGE_BYTES: u32 = 16 * 1024 * 1024;
pub const NATIVE_FFI_HOST_INLINE_VALUE_BYTES: usize = 128 * 1024;
pub const NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES: u64 = 512 * 1024 * 1024;
pub const NATIVE_FFI_HOST_MAX_ARGUMENTS: usize = 4096;
pub const NATIVE_FFI_HOST_MAX_OUTPUTS: usize = 4096;
pub const NATIVE_FFI_HOST_MAX_CALLBACK_DEPTH: u16 = 64;
pub const NATIVE_FFI_HOST_MAX_RELEASES: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    rename_all = "snake_case",
    tag = "storage",
    content = "value"
)]
pub enum NativeValueTransfer {
    Inline(ValuePayload),
    Snapshot(SharedMemoryDescriptor),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRemoteReference {
    pub id: u64,
    pub type_identity: ForeignTypeIdentity,
    pub ownership: ForeignOwnership,
    pub affinity: ForeignAffinity,
    pub lifetime: ForeignLifetime,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    rename_all = "snake_case",
    tag = "kind",
    content = "value"
)]
pub enum NativeWireValue {
    Portable(NativeValueTransfer),
    Foreign(NativeRemoteReference),
    Callback { id: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    rename_all = "snake_case",
    tag = "operation",
    content = "payload"
)]
pub enum NativeDriverMessage {
    Invoke(NativeInvocationRequest),
    ProjectLoad(NativeProjectLoadRequest),
    CallbackResult(NativeCallbackResult),
    Shutdown { request_id: u64, releases: Vec<u64> },
}

/// One parent-issued, exact project load into a new isolated child. Paths are
/// relative to the authenticated private root owned by the parent lease.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProjectLoadRequest {
    pub request_id: u64,
    pub private_root: std::path::PathBuf,
    pub library: std::path::PathBuf,
    pub manifest: std::path::PathBuf,
    pub alias: String,
    pub artifact_identity: runmat_types::ForeignArtifactIdentity,
    pub library_digest: Digest,
    pub manifest_digest: Digest,
}

impl NativeProjectLoadRequest {
    pub fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        if !self.private_root.is_absolute() {
            return Err("native project private root is not absolute".into());
        }
        for path in [&self.private_root, &self.library, &self.manifest] {
            if path.to_str().is_none_or(|text| text.len() > 4096) {
                return Err("native project path exceeds the protocol bound".into());
            }
        }
        for relative in [&self.library, &self.manifest] {
            if relative.as_os_str().is_empty()
                || !relative
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
            {
                return Err("native project load has an invalid relative path".into());
            }
        }
        validate_text(&self.alias, "project library alias")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    rename_all = "snake_case",
    tag = "event",
    content = "payload"
)]
pub enum NativeHostMessage {
    Invocation(NativeInvocationResult),
    Callback(NativeCallbackRequest),
    Shutdown(NativeShutdownResult),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInvocationRequest {
    pub request_id: u64,
    pub operation: String,
    pub arguments: Vec<NativeWireValue>,
    pub requested_outputs: u32,
    pub releases: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInvocationResult {
    pub request_id: u64,
    pub outcome: Result<Vec<NativeWireValue>, NativeWireError>,
    /// Child-owned inventory after this exact completed operation, including
    /// failures that nevertheless changed native-library state.
    pub user_state_present: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCallbackRequest {
    pub request_id: u64,
    pub callback_id: u64,
    pub depth: u16,
    pub arguments: Vec<NativeWireValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCallbackResult {
    pub request_id: u64,
    pub callback_id: u64,
    pub outcome: Result<Vec<NativeWireValue>, NativeWireError>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeShutdownResult {
    pub request_id: u64,
    pub outcome: Result<(), NativeWireError>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeWireError {
    pub identifier: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cause: Option<NativeWireCause>,
}

impl NativeDriverMessage {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Invoke(request) => request.validate(),
            Self::ProjectLoad(request) => request.validate(),
            Self::CallbackResult(result) => result.validate(),
            Self::Shutdown {
                request_id,
                releases,
            } => {
                validate_id(*request_id, "request")?;
                validate_releases(releases)
            }
        }
    }
}

impl NativeHostMessage {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Invocation(result) => result.validate(),
            Self::Callback(request) => request.validate(),
            Self::Shutdown(result) => result.validate(),
        }
    }
}

impl NativeInvocationRequest {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        validate_text(&self.operation, "operation")?;
        if self.arguments.len() > NATIVE_FFI_HOST_MAX_ARGUMENTS {
            return Err("native FFI argument count exceeds the protocol limit".into());
        }
        if self.requested_outputs as usize > NATIVE_FFI_HOST_MAX_OUTPUTS {
            return Err("native FFI output count exceeds the protocol limit".into());
        }
        validate_releases(&self.releases)?;
        for value in &self.arguments {
            value.validate(0)?;
        }
        Ok(())
    }
}

impl NativeInvocationResult {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        match &self.outcome {
            Ok(values) => validate_outputs(values),
            Err(error) => error.validate(),
        }
    }
}

impl NativeCallbackRequest {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        validate_id(self.callback_id, "callback")?;
        if self.depth == 0 || self.depth > NATIVE_FFI_HOST_MAX_CALLBACK_DEPTH {
            return Err("native FFI callback depth exceeds the protocol limit".into());
        }
        if self.arguments.len() > NATIVE_FFI_HOST_MAX_ARGUMENTS {
            return Err("native FFI callback argument count exceeds the protocol limit".into());
        }
        for value in &self.arguments {
            value.validate(0)?;
        }
        Ok(())
    }
}

impl NativeCallbackResult {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        validate_id(self.callback_id, "callback")?;
        match &self.outcome {
            Ok(values) => validate_outputs(values),
            Err(error) => error.validate(),
        }
    }
}

impl NativeShutdownResult {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        match &self.outcome {
            Ok(()) => Ok(()),
            Err(error) => error.validate(),
        }
    }
}

impl NativeWireError {
    pub(super) fn from_runtime(error: crate::RuntimeError) -> Self {
        Self::from_runtime_ref(&error)
    }

    pub(super) fn from_runtime_ref(error: &crate::RuntimeError) -> Self {
        Self {
            identifier: error
                .identifier()
                .unwrap_or("RunMat:NativeFFI:InvocationFailed")
                .into(),
            message: error.to_string(),
            cause: NativeWireCause::from_runtime(error),
        }
    }

    pub(super) fn into_runtime(self) -> crate::RuntimeError {
        let error = crate::build_runtime_error(self.message)
            .with_builtin("native_ffi")
            .with_identifier(self.identifier);
        match self.cause {
            Some(cause) => error.with_source(cause.into_source()).build(),
            None => error.build(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_text(&self.identifier, "error identifier")?;
        validate_text(&self.message, "error message")?;
        if let Some(cause) = &self.cause {
            cause.validate()?;
        }
        Ok(())
    }
}

impl NativeWireValue {
    fn validate(&self, depth: u16) -> Result<(), String> {
        if depth >= NATIVE_FFI_HOST_MAX_CALLBACK_DEPTH {
            return Err("native FFI value nesting exceeds the protocol limit".into());
        }
        match self {
            Self::Portable(NativeValueTransfer::Inline(value)) => value
                .validate(runmat_execution::value::ValueLimits {
                    max_inline_bytes: NATIVE_FFI_HOST_INLINE_VALUE_BYTES as u64,
                    ..Default::default()
                })
                .map_err(|error| error.to_string()),
            Self::Portable(NativeValueTransfer::Snapshot(descriptor)) => {
                descriptor.validate().map_err(|error| error.to_string())?;
                if descriptor.byte_length > NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES {
                    return Err("native FFI value snapshot exceeds the protocol limit".into());
                }
                Ok(())
            }
            Self::Foreign(reference) => {
                validate_id(reference.id, "foreign resource")?;
                validate_text(&reference.type_identity.family, "foreign type family")?;
                validate_text(&reference.type_identity.name, "foreign type name")
            }
            Self::Callback { id } => validate_id(*id, "callback"),
        }
    }
}

fn validate_outputs(values: &[NativeWireValue]) -> Result<(), String> {
    if values.len() > NATIVE_FFI_HOST_MAX_OUTPUTS {
        return Err("native FFI output count exceeds the protocol limit".into());
    }
    values.iter().try_for_each(|value| value.validate(0))
}

fn validate_releases(releases: &[u64]) -> Result<(), String> {
    if releases.len() > NATIVE_FFI_HOST_MAX_RELEASES {
        return Err("native FFI release batch exceeds the protocol limit".into());
    }
    for release in releases {
        validate_id(*release, "released resource")?;
    }
    Ok(())
}

fn validate_id(id: u64, label: &str) -> Result<(), String> {
    if id == 0 {
        Err(format!("{label} id must be nonzero"))
    } else {
        Ok(())
    }
}

fn validate_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    if value.len() > NATIVE_FFI_HOST_MAX_MESSAGE_BYTES as usize {
        return Err(format!("{label} exceeds the protocol limit"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_error_keeps_origin_identifier_through_wire_roundtrip() {
        let original = crate::build_runtime_error("callback control")
            .with_identifier("RunMat:NativeFFI:OriginControl")
            .build();
        let result = NativeCallbackResult {
            request_id: 1,
            callback_id: 2,
            outcome: Err(NativeWireError::from_runtime(original)),
        };
        result.validate().unwrap();
        let encoded = serde_json::to_vec(&result).unwrap();
        let decoded: NativeCallbackResult = serde_json::from_slice(&encoded).unwrap();
        let error = decoded.outcome.unwrap_err().into_runtime();
        assert_eq!(error.identifier(), Some("RunMat:NativeFFI:OriginControl"));
        assert!(error.to_string().contains("callback control"));
    }

    #[test]
    fn protocol_rejects_zero_identities_and_oversized_batches() {
        let zero = NativeDriverMessage::Shutdown {
            request_id: 0,
            releases: Vec::new(),
        };
        assert!(zero.validate().is_err());
        let oversized = NativeDriverMessage::Shutdown {
            request_id: 1,
            releases: vec![1; NATIVE_FFI_HOST_MAX_RELEASES + 1],
        };
        assert!(oversized.validate().is_err());
    }

    #[test]
    fn protocol_rejects_unknown_fields() {
        let encoded =
            r#"{"operation":"shutdown","payload":{"request_id":1,"releases":[],"extra":true}}"#;
        assert!(serde_json::from_str::<NativeDriverMessage>(encoded).is_err());
    }

    #[test]
    fn frozen_v1_nested_output_list_is_not_a_v3_wire_value() {
        assert_eq!(NATIVE_FFI_HOST_SCHEMA_V1, 1);
        assert_eq!(NATIVE_FFI_HOST_SCHEMA_VERSION, 3);
        let frozen = r#"{"kind":"output_list","value":[]}"#;
        let current = runmat_process_host::ipc::HostHandshake::new(
            NATIVE_FFI_HOST_PROTOCOL,
            NATIVE_FFI_HOST_SCHEMA_VERSION,
            NATIVE_FFI_HOST_MAX_MESSAGE_BYTES,
        );
        let stale = runmat_process_host::ipc::HostHandshake::new(
            NATIVE_FFI_HOST_PROTOCOL,
            NATIVE_FFI_HOST_SCHEMA_V1,
            NATIVE_FFI_HOST_MAX_MESSAGE_BYTES,
        );
        assert!(runmat_process_host::ipc::negotiate_handshake(&current, &stale).is_err());
        let prior = runmat_process_host::ipc::HostHandshake::new(
            NATIVE_FFI_HOST_PROTOCOL,
            NATIVE_FFI_HOST_SCHEMA_V2,
            NATIVE_FFI_HOST_MAX_MESSAGE_BYTES,
        );
        assert!(runmat_process_host::ipc::negotiate_handshake(&current, &prior).is_err());
        assert!(serde_json::from_str::<NativeWireValue>(frozen).is_err());
    }

    #[test]
    fn top_level_output_vectors_are_bounded() {
        let result = NativeInvocationResult {
            request_id: 1,
            user_state_present: false,
            outcome: Ok(vec![
                NativeWireValue::Callback { id: 1 };
                NATIVE_FFI_HOST_MAX_OUTPUTS + 1
            ]),
        };
        assert!(result.validate().is_err());
    }

    #[test]
    fn current_top_level_output_vectors_preserve_zero_one_and_many_values() {
        for count in [0, 1, 3] {
            let result = NativeInvocationResult {
                request_id: 1,
                user_state_present: false,
                outcome: Ok((1..=count)
                    .map(|id| NativeWireValue::Callback { id })
                    .collect()),
            };
            result.validate().expect("valid output vector");
            let encoded = serde_json::to_vec(&result).expect("encode output vector");
            let decoded: NativeInvocationResult =
                serde_json::from_slice(&encoded).expect("decode output vector");
            assert_eq!(decoded, result);
        }
    }

    #[test]
    fn shutdown_error_keeps_request_identity_and_typed_wire_cause() {
        let result = NativeHostMessage::Shutdown(NativeShutdownResult {
            request_id: 17,
            outcome: Err(NativeWireError {
                identifier: "RunMat:onCleanup:CallbackFailed".into(),
                message: "cleanup failed".into(),
                cause: None,
            }),
        });
        result.validate().expect("valid failed shutdown envelope");
        let encoded = serde_json::to_vec(&result).expect("encode failed shutdown");
        let decoded: NativeHostMessage = serde_json::from_slice(&encoded).expect("decode shutdown");
        assert_eq!(decoded, result);
    }
}
