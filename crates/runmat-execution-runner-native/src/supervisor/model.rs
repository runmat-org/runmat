use std::path::{Path, PathBuf};

use runmat_execution::value::{ValueLimits, ValuePayload};
use runmat_execution::{Digest, JobHandle};
use runmat_execution_artifact::{
    ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest, ProgramExecutionResponse,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use serde::{Deserialize, Serialize};

use crate::{NativeExecutionError, NativeExecutionResult};

pub const MAX_BATCH_SOURCE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_BATCH_ARGUMENTS: usize = 4096;
pub const MAX_BATCH_ARGUMENT_BYTES: usize = 1024 * 1024;
pub const MIN_RETENTION_MILLIS: u64 = 60_000;
pub const MAX_RETENTION_MILLIS: u64 = 90 * 24 * 60 * 60 * 1000;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSubmission {
    pub source_name: String,
    pub source: Vec<u8>,
    pub arguments: Vec<String>,
    pub working_directory: String,
    pub idempotency_key: Option<String>,
    pub retention_millis: u64,
}

impl BatchSubmission {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        if self.source.is_empty() || self.source.len() > MAX_BATCH_SOURCE_BYTES {
            return Err(NativeExecutionError::Configuration(
                "batch source is empty or exceeds the 8 MiB local-control limit".into(),
            ));
        }
        if !valid_source_name(&self.source_name)
            || self.arguments.len() > MAX_BATCH_ARGUMENTS
            || self.arguments.iter().map(String::len).sum::<usize>() > MAX_BATCH_ARGUMENT_BYTES
            || self
                .arguments
                .iter()
                .any(|argument| argument.chars().any(|character| character == '\0'))
            || self
                .idempotency_key
                .as_deref()
                .is_some_and(|key| key.is_empty() || key.len() > 256 || key.contains('\0'))
            || self.retention_millis < MIN_RETENTION_MILLIS
            || self.retention_millis > MAX_RETENTION_MILLIS
        {
            return Err(NativeExecutionError::Configuration(
                "batch submission contains an invalid name, argument, key, or retention".into(),
            ));
        }
        let working_directory = Path::new(&self.working_directory);
        if !working_directory.is_absolute() || !working_directory.is_dir() {
            return Err(NativeExecutionError::Configuration(
                "batch working directory must be an existing absolute directory".into(),
            ));
        }
        Ok(())
    }

    pub fn request_digest(&self) -> NativeExecutionResult<Digest> {
        let mut identity = b"runmat-local-batch-request-v1\0".to_vec();
        let mut stable = self.clone();
        stable.idempotency_key = None;
        identity.extend(
            serde_json::to_vec(&stable)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?,
        );
        Ok(Digest::sha256(identity))
    }

    pub fn idempotency_digest(&self) -> Option<Digest> {
        self.idempotency_key.as_deref().map(|key| {
            Digest::sha256([b"runmat-local-batch-idempotency-v1\0", key.as_bytes()].concat())
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramBatchSubmission {
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
    pub function: usize,
    pub arguments: Vec<ValuePayload>,
    pub requested_outputs: u16,
    pub idempotency_key: Option<String>,
    pub retention_millis: u64,
}

impl ProgramBatchSubmission {
    pub fn from_request(
        request: ProgramExecutionRequest,
        idempotency_key: Option<String>,
        retention_millis: u64,
    ) -> Self {
        Self {
            recipe: request.recipe,
            artifact: request.artifact,
            function: request.function,
            arguments: request.arguments,
            requested_outputs: request.requested_outputs,
            idempotency_key,
            retention_millis,
        }
    }

    pub fn program_request(&self) -> ProgramExecutionRequest {
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe: self.recipe.clone(),
            artifact: self.artifact.clone(),
            function: self.function,
            arguments: self.arguments.clone(),
            requested_outputs: self.requested_outputs,
        }
    }

    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.program_request()
            .validate_for_portable_host()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        if self.arguments.len() > MAX_BATCH_ARGUMENTS
            || self
                .idempotency_key
                .as_deref()
                .is_some_and(|key| key.is_empty() || key.len() > 256 || key.contains('\0'))
            || self.retention_millis < MIN_RETENTION_MILLIS
            || self.retention_millis > MAX_RETENTION_MILLIS
        {
            return Err(NativeExecutionError::Configuration(
                "program batch submission contains an inconsistent callable/output contract or invalid argument, key, or retention".into(),
            ));
        }
        for argument in &self.arguments {
            argument
                .validate(ValueLimits::default())
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        }
        Ok(())
    }

    pub fn request_digest(&self) -> NativeExecutionResult<Digest> {
        let mut stable = self.clone();
        stable.idempotency_key = None;
        let mut identity = b"runmat-local-program-batch-request-v1\0".to_vec();
        identity.extend(
            serde_json::to_vec(&stable)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?,
        );
        Ok(Digest::sha256(identity))
    }

    pub fn idempotency_digest(&self) -> Option<Digest> {
        self.idempotency_key.as_deref().map(idempotency_digest)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalJobState {
    Queued,
    Starting,
    Running,
    Cancelling,
    Succeeded,
    Failed,
    Cancelled,
    Indeterminate,
}

impl LocalJobState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Indeterminate
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalJobRecord {
    pub schema_version: u16,
    pub handle: JobHandle,
    pub state: LocalJobState,
    pub request_digest: Digest,
    pub workload_digest: Digest,
    pub idempotency_key_digest: Option<Digest>,
    pub submitted_unix_millis: u64,
    pub updated_unix_millis: u64,
    pub retain_until_unix_millis: u64,
    pub driver_process_id: Option<u32>,
    pub exit_code: Option<i32>,
    pub message: Option<String>,
}

impl LocalJobRecord {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 1
            || self.handle.generation == 0
            || self.updated_unix_millis < self.submitted_unix_millis
            || self.retain_until_unix_millis < self.updated_unix_millis
            || matches!(self.state, LocalJobState::Running) && self.driver_process_id.is_none()
            || !matches!(
                self.state,
                LocalJobState::Running | LocalJobState::Cancelling
            ) && self.driver_process_id.is_some()
        {
            return Err(NativeExecutionError::Protocol(
                "durable local job record is inconsistent".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverCompletion {
    pub schema_version: u16,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub message: Option<String>,
    pub response: Option<ProgramExecutionResponse>,
}

impl DriverCompletion {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        let response_succeeded = self.response.as_ref().map(|response| {
            matches!(
                response,
                ProgramExecutionResponse::Success { .. }
                    | ProgramExecutionResponse::ExternalizedSuccess { .. }
            )
        });
        if self.schema_version != 2
            || self
                .message
                .as_ref()
                .is_some_and(|value| value.len() > 16 * 1024)
            || response_succeeded.is_some_and(|succeeded| succeeded != self.success)
        {
            return Err(NativeExecutionError::Protocol(
                "durable driver completion is malformed".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverMarker {
    pub schema_version: u16,
    pub process_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobAttachment {
    pub record: LocalJobRecord,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub next_stdout_offset: u64,
    pub next_stderr_offset: u64,
    pub response: Option<ProgramExecutionResponse>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchDriverInvocation {
    Script {
        job_directory: PathBuf,
        source_path: PathBuf,
        arguments: Vec<String>,
        working_directory: PathBuf,
    },
    Program {
        job_directory: PathBuf,
        submission: Box<ProgramBatchSubmission>,
    },
}

impl BatchDriverInvocation {
    pub fn job_directory(&self) -> &Path {
        match self {
            Self::Script { job_directory, .. } | Self::Program { job_directory, .. } => {
                job_directory
            }
        }
    }
}

fn valid_source_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value.is_ascii()
        && !value
            .chars()
            .any(|character| matches!(character, '/' | '\\' | '\0'))
        && value != "."
        && value != ".."
        && value.to_ascii_lowercase().ends_with(".m")
}

fn idempotency_digest(key: &str) -> Digest {
    Digest::sha256([b"runmat-local-batch-idempotency-v1\0", key.as_bytes()].concat())
}
