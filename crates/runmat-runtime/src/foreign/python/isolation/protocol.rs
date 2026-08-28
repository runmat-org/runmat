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
pub const PYTHON_HOST_SCHEMA_VERSION: u16 = 1;
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
    OutputList(Vec<PythonWireValue>),
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
    pub outcome: Result<PythonWireValue, PythonWireError>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonCallbackRequest {
    pub request_id: u64,
    pub callback_id: u64,
    pub depth: u16,
    pub arguments: Vec<PythonWireValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonCallbackResult {
    pub request_id: u64,
    pub callback_id: u64,
    pub outcome: Result<PythonWireValue, PythonWireError>,
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
            Ok(value) => value.validate(0),
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
            Ok(value) => value.validate(0),
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
            Self::OutputList(values) => values
                .iter()
                .try_for_each(|value| value.validate(depth + 1)),
        }
    }
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
