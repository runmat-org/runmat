use std::path::{Path, PathBuf};

use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, JobHandle};
use runmat_execution_artifact::{
    ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest, ProgramExecutionResponse,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V3,
};
use runmat_execution_runner::AttemptResourceAccounting;
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
        identity.extend(runmat_canonical_codec::encode_json_bounded(
            &stable,
            u64::from(super::protocol::SUPERVISOR_MAX_MESSAGE_BYTES),
        )?);
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
pub struct ProgramBatchSubmission<
    R = ProgramBuildRecipe,
    A = ProgramArtifact,
    V = Vec<ValuePayload>,
    T = runmat_execution_runner::TaskSubmission,
    E = runmat_execution::EnvironmentSnapshot,
> {
    pub recipe: R,
    pub artifact: A,
    pub function: usize,
    pub arguments: V,
    pub environment: E,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_controls")]
    pub console: Option<runmat_execution::console::ConsoleLimits>,
    pub requested_outputs: u16,
    pub task: T,
    pub idempotency_key: Option<String>,
    pub retention_millis: u64,
}

impl ProgramBatchSubmission {
    pub fn from_request(
        request: ProgramExecutionRequest,
        task: runmat_execution_runner::TaskSubmission,
        idempotency_key: Option<String>,
        retention_millis: u64,
    ) -> Self {
        Self {
            recipe: request.recipe,
            artifact: request.artifact,
            function: request.function,
            arguments: request.arguments,
            environment: request.environment,
            console: request.console,
            requested_outputs: request.requested_outputs,
            task,
            idempotency_key,
            retention_millis,
        }
    }

    pub fn program_request(
        &self,
    ) -> runmat_execution_artifact::BorrowedProgramExecutionRequest<'_> {
        ProgramExecutionRequest {
            console: self.console,
            environment: &self.environment,
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V3,
            recipe: &self.recipe,
            artifact: &self.artifact,
            function: self.function,
            arguments: self.arguments.as_slice(),
            requested_outputs: self.requested_outputs,
        }
    }

    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.program_request()
            .validate_for_portable_host()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        self.task
            .request
            .validate_invocation_reservation()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        self.task
            .request
            .resources
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        if self.task.request.console != self.console
            || self.task.request.environment != self.environment
            || self.task.request.inputs != self.arguments
            || self.task.request.outputs.requested_outputs != self.requested_outputs
            || self.task.request.program_artifact_id
                != runmat_execution::identity::ArtifactId::derive(&[self.artifact.id.0.bytes()])
        {
            return Err(NativeExecutionError::Protocol(
                "durable task differs from its original program invocation".into(),
            ));
        }
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
        Ok(())
    }

    pub fn request_digest(&self) -> NativeExecutionResult<Digest> {
        // Same schema and identity bytes; borrowed physics/program payloads are not
        // copied merely to omit the caller's idempotency key from content identity.
        let stable = ProgramBatchSubmission {
            recipe: &self.recipe,
            artifact: &self.artifact,
            function: self.function,
            arguments: self.arguments.as_slice(),
            environment: &self.environment,
            console: self.console,
            requested_outputs: self.requested_outputs,
            task: &self.task,
            idempotency_key: None,
            retention_millis: self.retention_millis,
        };
        let mut identity = b"runmat-local-program-batch-request-v1\0".to_vec();
        identity.extend(runmat_canonical_codec::encode_json_bounded(
            &stable,
            u64::from(super::protocol::SUPERVISOR_MAX_MESSAGE_BYTES),
        )?);
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
}

impl DriverCompletion {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 4
            || self
                .message
                .as_ref()
                .is_some_and(|value| value.len() > 16 * 1024)
            || self.success && self.exit_code != Some(0)
        {
            return Err(NativeExecutionError::Protocol(
                "durable script completion is malformed".into(),
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
    #[serde(deserialize_with = "required_marker_attempt")]
    pub attempt_id: Option<runmat_execution::identity::AttemptId>,
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
    pub resources: Option<AttemptResourceAccounting>,
    pub execution: Option<runmat_execution_runner::TaskExecutionEvidence>,
    pub latest_progress: Option<crate::protocol::ProgramProgress>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramBatchExecution {
    pub response: ProgramExecutionResponse,
    pub report: runmat_execution_runner::port::BackendReport,
    pub latest_progress: Option<crate::protocol::ProgramProgress>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableProgramAttempt {
    pub schema_version: u16,
    pub request: runmat_execution_runner::AttemptRequest,
    pub config: crate::NativeExecutionConfig,
    pub job_directory: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableExecutionContext {
    pub scope_id: runmat_execution::ExecutionScopeId,
    pub pool_id: runmat_execution::PoolId,
}

#[derive(Debug)]
pub enum BatchDriverInvocation {
    Script {
        job_directory: PathBuf,
        source_path: PathBuf,
        arguments: Vec<String>,
        working_directory: PathBuf,
    },
    Program {
        job_directory: PathBuf,
        attempt: Box<DurableProgramAttempt>,
        submission: Box<ProgramBatchSubmission>,
        lease: Box<super::DurableAttemptLease>,
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

pub(super) fn valid_source_name(value: &str) -> bool {
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

fn required_marker_attempt<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<runmat_execution::identity::AttemptId>, D::Error> {
    Option::<runmat_execution::identity::AttemptId>::deserialize(deserializer)
}
