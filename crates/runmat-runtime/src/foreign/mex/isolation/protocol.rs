use runmat_execution::value::ValuePayload;
use runmat_mex::MexApi;
use runmat_process_host::shared_memory::SharedMemoryDescriptor;
use serde::{Deserialize, Serialize};

pub const MEX_HOST_PROTOCOL: &str = "runmat.mex-host";
pub const MEX_HOST_SECRET_ENV: &str = "RUNMAT_EXTENSION_HOST_SECRET";
pub const MEX_HOST_SNAPSHOT_ROOT_ENV: &str = "RUNMAT_EXTENSION_SNAPSHOT_ROOT";
pub const MEX_HOST_SCHEMA_VERSION: u16 = 1;
pub const MEX_HOST_MAX_MESSAGE_BYTES: u32 = 16 * 1024 * 1024;
pub const MEX_HOST_INLINE_VALUE_BYTES: usize = 128 * 1024;
pub const MEX_HOST_MAX_SNAPSHOT_BYTES: u64 = 512 * 1024 * 1024;
pub const MEX_HOST_MAX_ARGUMENTS: usize = 4096;
pub const MEX_HOST_MAX_OUTPUTS: usize = 4096;
pub const MEX_HOST_MAX_CALLBACK_DEPTH: u16 = 64;
pub const MEX_HOST_MAX_TEXT_BYTES: usize = MEX_HOST_MAX_MESSAGE_BYTES as usize;
pub const MEX_HOST_MAX_DIAGNOSTICS: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MexBinaryTier {
    RunMatExact,
    RunMatCompatibleIsolated,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "storage", content = "value")]
pub enum MexValueTransfer {
    Inline(ValuePayload),
    Snapshot(SharedMemoryDescriptor),
}

impl MexValueTransfer {
    pub(super) fn validate(&self) -> Result<(), MexProtocolError> {
        match self {
            Self::Inline(value) => value
                .validate(runmat_execution::value::ValueLimits {
                    max_inline_bytes: MEX_HOST_INLINE_VALUE_BYTES as u64,
                    ..Default::default()
                })
                .map_err(|error| MexProtocolError::new(error.to_string())),
            Self::Snapshot(descriptor) => {
                descriptor
                    .validate()
                    .map_err(|error| MexProtocolError::new(error.to_string()))?;
                if descriptor.byte_length > MEX_HOST_MAX_SNAPSHOT_BYTES {
                    return Err(MexProtocolError::new(
                        "MEX value snapshot exceeds the protocol limit",
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexInvocationRequest {
    pub request_id: u64,
    pub module_path: String,
    pub tier: MexBinaryTier,
    pub api: Option<MexApi>,
    pub arguments: Vec<MexValueTransfer>,
    pub requested_outputs: u32,
}

impl MexInvocationRequest {
    pub fn validate(&self) -> Result<(), MexProtocolError> {
        if self.request_id == 0 {
            return Err(MexProtocolError::new("request id must be nonzero"));
        }
        validate_path(&self.module_path)?;
        if self.arguments.len() > MEX_HOST_MAX_ARGUMENTS {
            return Err(MexProtocolError::new(
                "MEX argument count exceeds the protocol limit",
            ));
        }
        if self.requested_outputs as usize > MEX_HOST_MAX_OUTPUTS {
            return Err(MexProtocolError::new(
                "MEX output count exceeds the protocol limit",
            ));
        }
        for argument in &self.arguments {
            argument.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation", content = "payload")]
pub enum DriverMessage {
    Invoke(MexInvocationRequest),
    Clear {
        request_id: u64,
        module_path: String,
    },
    Shutdown {
        request_id: u64,
    },
    CallbackResult(MexCallbackResult),
}

impl DriverMessage {
    pub fn validate(&self) -> Result<(), MexProtocolError> {
        match self {
            Self::Invoke(request) => request.validate(),
            Self::Clear {
                request_id,
                module_path,
            } => {
                validate_identity(*request_id, "request")?;
                validate_path(module_path)
            }
            Self::Shutdown { request_id } => validate_identity(*request_id, "request"),
            Self::CallbackResult(result) => result.validate(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "event", content = "payload")]
pub enum HostMessage {
    Invocation(MexInvocationResult),
    Lifecycle(MexLifecycleResult),
    Callback(MexCallbackRequest),
}

impl HostMessage {
    pub fn validate(&self) -> Result<(), MexProtocolError> {
        match self {
            Self::Invocation(result) => result.validate(),
            Self::Lifecycle(result) => result.validate(),
            Self::Callback(request) => request.validate(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexInvocationResult {
    pub request_id: u64,
    pub outcome: Result<MexInvocationOutput, MexWireError>,
}

impl MexInvocationResult {
    fn validate(&self) -> Result<(), MexProtocolError> {
        validate_identity(self.request_id, "request")?;
        match &self.outcome {
            Ok(output) => output.validate(),
            Err(error) => error.validate(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexInvocationOutput {
    pub outputs: Vec<MexValueTransfer>,
    pub warnings: Vec<MexWireDiagnostic>,
    pub console: String,
}

impl MexInvocationOutput {
    fn validate(&self) -> Result<(), MexProtocolError> {
        if self.outputs.len() > MEX_HOST_MAX_OUTPUTS {
            return Err(MexProtocolError::new(
                "MEX output count exceeds the protocol limit",
            ));
        }
        for output in &self.outputs {
            output.validate()?;
        }
        if self.warnings.len() > MEX_HOST_MAX_DIAGNOSTICS {
            return Err(MexProtocolError::new(
                "MEX diagnostic count exceeds the protocol limit",
            ));
        }
        for warning in &self.warnings {
            warning.validate()?;
        }
        validate_text(&self.console, "console output")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexLifecycleResult {
    pub request_id: u64,
    pub outcome: Result<MexLifecycleOutcome, MexWireError>,
}

impl MexLifecycleResult {
    fn validate(&self) -> Result<(), MexProtocolError> {
        validate_identity(self.request_id, "request")?;
        if let Err(error) = &self.outcome {
            error.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MexLifecycleOutcome {
    Cleared,
    Retained,
    Shutdown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexCallbackRequest {
    pub request_id: u64,
    pub callback_id: u64,
    pub depth: u16,
    pub operation: MexCallbackOperation,
}

impl MexCallbackRequest {
    pub fn validate(&self) -> Result<(), MexProtocolError> {
        validate_identity(self.request_id, "request")?;
        validate_identity(self.callback_id, "callback")?;
        if self.depth == 0 || self.depth > MEX_HOST_MAX_CALLBACK_DEPTH {
            return Err(MexProtocolError::new(
                "callback depth exceeds the protocol limit",
            ));
        }
        self.operation.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation", content = "payload")]
pub enum MexCallbackOperation {
    Eval {
        command: String,
    },
    Call {
        function: String,
        arguments: Vec<MexValueTransfer>,
        requested_outputs: u32,
    },
    GetVariable {
        workspace: String,
        name: String,
    },
    PutVariable {
        workspace: String,
        name: String,
        value: MexValueTransfer,
    },
}

impl MexCallbackOperation {
    fn validate(&self) -> Result<(), MexProtocolError> {
        match self {
            Self::Eval { command } => validate_text(command, "evaluation command"),
            Self::Call {
                function,
                arguments,
                requested_outputs,
            } => {
                validate_name(function, "function")?;
                if arguments.len() > MEX_HOST_MAX_ARGUMENTS {
                    return Err(MexProtocolError::new(
                        "MEX callback argument count exceeds the protocol limit",
                    ));
                }
                if *requested_outputs as usize > MEX_HOST_MAX_OUTPUTS {
                    return Err(MexProtocolError::new(
                        "MEX callback output count exceeds the protocol limit",
                    ));
                }
                for argument in arguments {
                    argument.validate()?;
                }
                Ok(())
            }
            Self::GetVariable { workspace, name } => {
                validate_name(workspace, "workspace")?;
                validate_name(name, "variable")
            }
            Self::PutVariable {
                workspace,
                name,
                value,
            } => {
                validate_name(workspace, "workspace")?;
                validate_name(name, "variable")?;
                value.validate()
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexCallbackResult {
    pub request_id: u64,
    pub callback_id: u64,
    pub outcome: Result<MexCallbackOutput, MexWireError>,
}

impl MexCallbackResult {
    fn validate(&self) -> Result<(), MexProtocolError> {
        validate_identity(self.request_id, "request")?;
        validate_identity(self.callback_id, "callback")?;
        match &self.outcome {
            Ok(output) => output.validate(),
            Err(error) => error.validate(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "result", content = "payload")]
pub enum MexCallbackOutput {
    Unit,
    Values(Vec<MexValueTransfer>),
    OptionalValue(Option<MexValueTransfer>),
}

impl MexCallbackOutput {
    fn validate(&self) -> Result<(), MexProtocolError> {
        match self {
            Self::Unit => Ok(()),
            Self::Values(values) => {
                if values.len() > MEX_HOST_MAX_OUTPUTS {
                    return Err(MexProtocolError::new(
                        "MEX callback value count exceeds the protocol limit",
                    ));
                }
                for value in values {
                    value.validate()?;
                }
                Ok(())
            }
            Self::OptionalValue(value) => {
                if let Some(value) = value {
                    value.validate()?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexWireDiagnostic {
    pub identifier: Option<String>,
    pub message: String,
}

impl MexWireDiagnostic {
    fn validate(&self) -> Result<(), MexProtocolError> {
        if let Some(identifier) = &self.identifier {
            validate_name(identifier, "diagnostic identifier")?;
        }
        validate_text(&self.message, "diagnostic message")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexWireError {
    pub identifier: String,
    pub message: String,
    pub dependency: Option<String>,
}

impl MexWireError {
    fn validate(&self) -> Result<(), MexProtocolError> {
        validate_name(&self.identifier, "error identifier")?;
        validate_text(&self.message, "error message")?;
        if let Some(dependency) = &self.dependency {
            validate_text(dependency, "dependency diagnostic")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid MEX host protocol record: {message}")]
pub struct MexProtocolError {
    message: String,
}

impl MexProtocolError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

fn validate_path(path: &str) -> Result<(), MexProtocolError> {
    if path.is_empty() || path.len() > 32 * 1024 || path.chars().any(char::is_control) {
        return Err(MexProtocolError::new("module path is invalid"));
    }
    Ok(())
}

fn validate_identity(identity: u64, kind: &str) -> Result<(), MexProtocolError> {
    if identity == 0 {
        Err(MexProtocolError::new(format!(
            "{kind} identity must be nonzero"
        )))
    } else {
        Ok(())
    }
}

fn validate_name(value: &str, kind: &str) -> Result<(), MexProtocolError> {
    if value.is_empty()
        || value.len() > 32 * 1024
        || value.chars().any(|character| character.is_control())
    {
        Err(MexProtocolError::new(format!("{kind} name is invalid")))
    } else {
        Ok(())
    }
}

fn validate_text(value: &str, kind: &str) -> Result<(), MexProtocolError> {
    if value.len() > MEX_HOST_MAX_TEXT_BYTES {
        Err(MexProtocolError::new(format!(
            "{kind} exceeds the protocol limit"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::value::InlineValue;
    use runmat_process_host::shared_memory::SharedMemoryKind;

    #[test]
    fn invocation_contract_round_trips_canonically() {
        let message = DriverMessage::Invoke(MexInvocationRequest {
            request_id: 7,
            module_path: "/tmp/add.mex".into(),
            tier: MexBinaryTier::RunMatExact,
            api: Some(MexApi::R2018a),
            arguments: vec![MexValueTransfer::Inline(ValuePayload::Inline(Box::new(
                InlineValue::U64(u64::MAX),
            )))],
            requested_outputs: 1,
        });
        let bytes = serde_json::to_vec(&message).unwrap();
        let decoded: DriverMessage = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, message);
        let DriverMessage::Invoke(request) = decoded else {
            panic!("expected invocation");
        };
        request.validate().unwrap();
    }

    #[test]
    fn protocol_rejects_unbounded_or_ambiguous_records() {
        let request = MexInvocationRequest {
            request_id: 0,
            module_path: String::new(),
            tier: MexBinaryTier::RunMatCompatibleIsolated,
            api: None,
            arguments: Vec::new(),
            requested_outputs: 0,
        };
        assert!(request.validate().is_err());

        let callback = MexCallbackRequest {
            request_id: 1,
            callback_id: 1,
            depth: MEX_HOST_MAX_CALLBACK_DEPTH + 1,
            operation: MexCallbackOperation::Eval {
                command: String::new(),
            },
        };
        assert!(callback.validate().is_err());

        let clear = DriverMessage::Clear {
            request_id: 0,
            module_path: "/tmp/a.mex".into(),
        };
        assert!(clear.validate().is_err());

        let invalid_snapshot = MexValueTransfer::Snapshot(SharedMemoryDescriptor {
            kind: SharedMemoryKind::FileBacked,
            name: "not-a-session-nonce".into(),
            byte_length: 1,
            nonce: [0; 16],
            sha256: [0; 32],
        });
        let host = HostMessage::Invocation(MexInvocationResult {
            request_id: 1,
            outcome: Ok(MexInvocationOutput {
                outputs: vec![invalid_snapshot],
                warnings: Vec::new(),
                console: String::new(),
            }),
        });
        assert!(host.validate().is_err());
    }

    #[test]
    fn unknown_fields_are_rejected_at_struct_boundaries() {
        let encoded = br#"{"request_id":1,"module_path":"/tmp/a.mex","tier":"run_mat_exact","api":"r2017b","arguments":[],"requested_outputs":0,"extra":true}"#;
        assert!(serde_json::from_slice::<MexInvocationRequest>(encoded).is_err());
    }
}
