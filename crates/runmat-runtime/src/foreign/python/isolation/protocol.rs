use runmat_execution::value::ValuePayload;
use runmat_process_host::shared_memory::SharedMemoryDescriptor;
use runmat_python::PythonError;
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use serde::{Deserialize, Serialize};

pub const PYTHON_HOST_KIND: &str = "python";
pub const PYTHON_HOST_KIND_ENV: &str = "RUNMAT_EXTENSION_HOST_KIND";
pub const PYTHON_HOST_CONFIG_ENV: &str = "RUNMAT_PYTHON_HOST_CONFIG";
pub const PYTHON_HOST_PROTOCOL: &str = "runmat.python-host";
pub const PYTHON_HOST_SECRET_ENV: &str = "RUNMAT_PYTHON_HOST_SECRET";
pub const PYTHON_HOST_SNAPSHOT_ROOT_ENV: &str = "RUNMAT_PYTHON_SNAPSHOT_ROOT";
#[cfg(test)]
pub const PYTHON_HOST_SCHEMA_V1: u16 = 1;
pub const PYTHON_HOST_SCHEMA_V2: u16 = 2;
pub const PYTHON_HOST_SCHEMA_VERSION: u16 = PYTHON_HOST_SCHEMA_V2;
pub const PYTHON_HOST_MAX_MESSAGE_BYTES: u32 = 16 * 1024 * 1024;
pub const PYTHON_HOST_INLINE_VALUE_BYTES: usize = 128 * 1024;
pub const PYTHON_HOST_MAX_SNAPSHOT_BYTES: u64 = 512 * 1024 * 1024;
pub const PYTHON_HOST_MAX_ARGUMENTS: usize = 4096;
pub const PYTHON_HOST_MAX_OUTPUTS: usize = 4096;
pub const PYTHON_HOST_MAX_CALLBACK_DEPTH: u16 = 64;
pub const PYTHON_HOST_MAX_RELEASES: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    rename_all = "snake_case",
    tag = "storage",
    content = "value"
)]
pub enum PythonValueTransfer {
    Inline(ValuePayload),
    Snapshot(SharedMemoryDescriptor),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonRemoteReference {
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
pub enum PythonWireValue {
    Portable(PythonValueTransfer),
    Foreign(PythonRemoteReference),
    Callback { id: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    rename_all = "snake_case",
    tag = "operation",
    content = "payload"
)]
pub enum PythonDriverMessage {
    Invoke(PythonInvocationRequest),
    CallbackResult(PythonCallbackResult),
    Shutdown { request_id: u64, releases: Vec<u64> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    rename_all = "snake_case",
    tag = "event",
    content = "payload"
)]
pub enum PythonHostMessage {
    Invocation(PythonInvocationResult),
    Callback(PythonCallbackRequest),
    Shutdown(PythonShutdownResult),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonInvocationRequest {
    pub request_id: u64,
    pub operation: String,
    pub arguments: Vec<PythonWireValue>,
    pub requested_outputs: u32,
    pub releases: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonInvocationResult {
    pub request_id: u64,
    pub outcome: Result<Vec<PythonWireValue>, PythonWireError>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonCallbackRequest {
    pub request_id: u64,
    pub callback_id: u64,
    pub depth: u16,
    pub requested_outputs: u32,
    pub arguments: Vec<PythonWireValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonCallbackResult {
    pub request_id: u64,
    pub callback_id: u64,
    pub outcome: Result<Vec<PythonWireValue>, PythonWireError>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonShutdownResult {
    pub request_id: u64,
    pub outcome: Result<(), PythonWireError>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonWireError {
    pub identifier: String,
    pub message: String,
    pub python: Option<PythonError>,
}

impl PythonDriverMessage {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Invoke(request) => request.validate(),
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

impl PythonHostMessage {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Invocation(result) => result.validate(),
            Self::Callback(request) => request.validate(),
            Self::Shutdown(result) => result.validate(),
        }
    }
}

impl PythonInvocationRequest {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        validate_text(&self.operation, "operation")?;
        if self.arguments.len() > PYTHON_HOST_MAX_ARGUMENTS {
            return Err("Python argument count exceeds the protocol limit".into());
        }
        if self.requested_outputs as usize > PYTHON_HOST_MAX_OUTPUTS {
            return Err("Python callback output count exceeds the protocol limit".into());
        }
        if self.requested_outputs as usize > PYTHON_HOST_MAX_OUTPUTS {
            return Err("Python output count exceeds the protocol limit".into());
        }
        validate_releases(&self.releases)?;
        self.arguments
            .iter()
            .try_for_each(|value| value.validate(0))
    }
}

impl PythonInvocationResult {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        match &self.outcome {
            Ok(values) => validate_outputs(values),
            Err(error) => error.validate(),
        }
    }
}

impl PythonCallbackRequest {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        validate_id(self.callback_id, "callback")?;
        if self.depth == 0 || self.depth > PYTHON_HOST_MAX_CALLBACK_DEPTH {
            return Err("Python callback depth exceeds the protocol limit".into());
        }
        if self.arguments.len() > PYTHON_HOST_MAX_ARGUMENTS {
            return Err("Python callback argument count exceeds the protocol limit".into());
        }
        self.arguments
            .iter()
            .try_for_each(|value| value.validate(0))
    }
}

impl PythonCallbackResult {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        validate_id(self.callback_id, "callback")?;
        match &self.outcome {
            Ok(values) => validate_outputs(values),
            Err(error) => error.validate(),
        }
    }
}

impl PythonShutdownResult {
    fn validate(&self) -> Result<(), String> {
        validate_id(self.request_id, "request")?;
        self.outcome
            .as_ref()
            .map(|_| ())
            .map_err(|error| error.message.clone())
    }
}

impl PythonWireError {
    pub fn validate(&self) -> Result<(), String> {
        validate_text(&self.identifier, "error identifier")?;
        validate_text(&self.message, "error message")
    }
}

impl PythonWireValue {
    fn validate(&self, depth: u16) -> Result<(), String> {
        if depth >= PYTHON_HOST_MAX_CALLBACK_DEPTH {
            return Err("Python value nesting exceeds the protocol limit".into());
        }
        match self {
            Self::Portable(PythonValueTransfer::Inline(value)) => value
                .validate(runmat_execution::value::ValueLimits {
                    max_inline_bytes: PYTHON_HOST_INLINE_VALUE_BYTES as u64,
                    ..Default::default()
                })
                .map_err(|error| error.to_string()),
            Self::Portable(PythonValueTransfer::Snapshot(descriptor)) => {
                if descriptor.byte_length > PYTHON_HOST_MAX_SNAPSHOT_BYTES {
                    Err("Python value snapshot exceeds the protocol limit".into())
                } else {
                    Ok(())
                }
            }
            Self::Foreign(reference) => validate_id(reference.id, "foreign resource"),
            Self::Callback { id } => validate_id(*id, "callback"),
        }
    }
}

fn validate_outputs(values: &[PythonWireValue]) -> Result<(), String> {
    if values.len() > PYTHON_HOST_MAX_OUTPUTS {
        return Err("Python output count exceeds the protocol limit".into());
    }
    values.iter().try_for_each(|value| value.validate(0))
}

fn validate_releases(releases: &[u64]) -> Result<(), String> {
    if releases.len() > PYTHON_HOST_MAX_RELEASES {
        return Err("Python release count exceeds the protocol limit".into());
    }
    releases
        .iter()
        .try_for_each(|id| validate_id(*id, "release"))
}

fn validate_id(id: u64, label: &str) -> Result<(), String> {
    if id == 0 {
        Err(format!("{label} identity must be nonzero"))
    } else {
        Ok(())
    }
}

fn validate_text(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 64 * 1024 || value.contains('\0') {
        Err(format!("{label} is invalid"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_v1_nested_output_list_is_not_a_v2_wire_value() {
        assert_eq!(PYTHON_HOST_SCHEMA_V1, 1);
        assert_eq!(PYTHON_HOST_SCHEMA_VERSION, 2);
        let frozen = r#"{"kind":"output_list","value":[]}"#;
        let current = runmat_process_host::ipc::HostHandshake::new(
            PYTHON_HOST_PROTOCOL,
            PYTHON_HOST_SCHEMA_VERSION,
            PYTHON_HOST_MAX_MESSAGE_BYTES,
        );
        let stale = runmat_process_host::ipc::HostHandshake::new(
            PYTHON_HOST_PROTOCOL,
            PYTHON_HOST_SCHEMA_V1,
            PYTHON_HOST_MAX_MESSAGE_BYTES,
        );
        assert!(runmat_process_host::ipc::negotiate_handshake(&current, &stale).is_err());
        assert!(serde_json::from_str::<PythonWireValue>(frozen).is_err());
    }

    #[test]
    fn top_level_output_vectors_are_bounded() {
        let result = PythonInvocationResult {
            request_id: 1,
            outcome: Ok(vec![
                PythonWireValue::Callback { id: 1 };
                PYTHON_HOST_MAX_OUTPUTS + 1
            ]),
        };
        assert!(result.validate().is_err());
    }

    #[test]
    fn current_top_level_output_vectors_preserve_zero_one_and_many_values() {
        for count in [0, 1, 3] {
            let result = PythonInvocationResult {
                request_id: 1,
                outcome: Ok((1..=count)
                    .map(|id| PythonWireValue::Callback { id })
                    .collect()),
            };
            result.validate().expect("valid output vector");
            let encoded = serde_json::to_vec(&result).expect("encode output vector");
            let decoded: PythonInvocationResult =
                serde_json::from_slice(&encoded).expect("decode output vector");
            assert_eq!(decoded, result);
        }
    }
}
