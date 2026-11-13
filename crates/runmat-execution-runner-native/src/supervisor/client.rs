use std::sync::Arc;
use std::time::Duration;

use runmat_execution::JobId;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use tokio::io::{AsyncRead, AsyncWrite};

use super::auth::read_token;
use super::model::{
    BatchSubmission, GraphBatchSubmission, JobAttachment, LocalJobRecord, ProgramBatchSubmission,
};
use super::protocol::{
    SupervisorCommand, SupervisorRequest, SupervisorResponse, SUPERVISOR_MAX_MESSAGE_BYTES,
    SUPERVISOR_PROTOCOL_VERSION,
};
use super::LocalSupervisorConfig;
use crate::{NativeExecutionError, NativeExecutionResult};

/// The supervisor's disposition, not a storage-layer "created" flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobSubmissionDisposition {
    Created,
    Duplicate,
}

#[derive(Clone)]
pub struct LocalSupervisorClient {
    config: Arc<LocalSupervisorConfig>,
    local_fea_organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
}

impl LocalSupervisorClient {
    pub fn new(config: LocalSupervisorConfig) -> Self {
        let selected =
            runmat_runtime::analysis::fea_run_admission::current_local_fea_organization_hint();
        let local_fea_organization = match selected {
            runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization::Default => {
                config.execution.local_fea_organization
            }
            selected => selected,
        };
        Self {
            config: Arc::new(config),
            local_fea_organization,
        }
    }

    pub fn for_current_executable() -> NativeExecutionResult<Self> {
        Ok(Self::new(LocalSupervisorConfig::for_current_executable()?))
    }

    pub async fn connect_or_start(&self) -> NativeExecutionResult<()> {
        match std::fs::symlink_metadata(&self.config.paths.token) {
            Ok(_) => {
                read_token(&self.config.paths.token)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(super::filesystem::io_error(error)),
        }
        match tokio::time::timeout(Duration::from_secs(5), self.ping())
            .await
            .map_err(|_| readiness_timeout())?
        {
            Ok(()) => return Ok(()),
            Err(error) if endpoint_absent(&error) => {}
            Err(error) => return Err(error),
        }
        self.start_original_supervisor().await
    }

    pub async fn ping(&self) -> NativeExecutionResult<()> {
        match self.request(SupervisorCommand::Ping).await? {
            SupervisorResponse::Pong { config_identity }
                if config_identity == self.config.identity()? =>
            {
                Ok(())
            }
            SupervisorResponse::Pong { .. } => Err(NativeExecutionError::Configuration(
                "local supervisor has a different execution profile".into(),
            )),
            response => unexpected(response),
        }
    }

    pub async fn execution_context(
        &self,
        idempotency_key: Option<&str>,
    ) -> NativeExecutionResult<super::DurableExecutionContext> {
        self.connect_or_start().await?;
        match self
            .request(SupervisorCommand::ExecutionContext {
                idempotency_key: idempotency_key.map(str::to_owned),
            })
            .await?
        {
            SupervisorResponse::ExecutionContext { context } => Ok(context),
            response => unexpected(response),
        }
    }

    pub async fn console_page(
        &self,
        handle: runmat_execution::JobHandle,
        attempt_id: runmat_execution::identity::AttemptId,
        cursor: runmat_execution::console::ConsoleCursor,
        budget: runmat_execution::console::ConsoleReadBudget,
    ) -> NativeExecutionResult<Option<runmat_execution::console::ConsolePage>> {
        budget
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        match self
            .request(SupervisorCommand::ConsolePage {
                handle,
                attempt_id,
                cursor,
                budget,
            })
            .await?
        {
            SupervisorResponse::ConsolePage {
                attempt_id: original,
                page,
            } if original == attempt_id => {
                if let Some(page) = &page {
                    page.validate()
                        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
                    if page.cursor != cursor || page.budget != budget {
                        return Err(NativeExecutionError::Protocol(
                            "console page differs from original read".into(),
                        ));
                    }
                }
                Ok(page.map(|page| *page))
            }
            response => unexpected(response),
        }
    }

    pub async fn submit(
        &self,
        submission: BatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, JobSubmissionDisposition)> {
        self.connect_or_start().await?;
        submitted(
            self.request(SupervisorCommand::Submit {
                submission: Box::new(submission),
            })
            .await?,
        )
    }

    pub async fn list(&self) -> NativeExecutionResult<Vec<LocalJobRecord>> {
        self.connect_or_start().await?;
        match self.request(SupervisorCommand::List).await? {
            SupervisorResponse::Jobs { records } => Ok(records),
            response => unexpected(response),
        }
    }

    pub async fn submit_program(
        &self,
        submission: ProgramBatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, JobSubmissionDisposition)> {
        self.connect_or_start().await?;
        submitted(
            self.request(SupervisorCommand::SubmitProgram {
                submission: Box::new(submission),
            })
            .await?,
        )
    }

    pub async fn submit_graph(
        &self,
        submission: GraphBatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, JobSubmissionDisposition)> {
        self.connect_or_start().await?;
        submitted(
            self.request(SupervisorCommand::SubmitGraph {
                submission: Box::new(submission),
            })
            .await?,
        )
    }

    pub async fn show(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        self.connect_or_start().await?;
        match self.request(SupervisorCommand::Show { job_id }).await? {
            SupervisorResponse::Job { record } => Ok(record),
            response => unexpected(response),
        }
    }

    pub async fn attach(
        &self,
        job_id: JobId,
        stdout_offset: u64,
        stderr_offset: u64,
    ) -> NativeExecutionResult<JobAttachment> {
        self.connect_or_start().await?;
        match self
            .request(SupervisorCommand::Attach {
                job_id,
                stdout_offset,
                stderr_offset,
            })
            .await?
        {
            SupervisorResponse::Attachment { attachment } => Ok(*attachment),
            response => unexpected(response),
        }
    }

    pub async fn raw_output_attempt(
        &self,
        handle: runmat_execution::JobHandle,
        ordinal: u32,
    ) -> NativeExecutionResult<super::RawProcessOutputLookup> {
        if ordinal == 0 || ordinal > u32::from(u16::MAX) + 1 {
            return Err(NativeExecutionError::Protocol(
                "raw output ordinal exceeds the attempt inventory".into(),
            ));
        }
        self.connect_or_start().await?;
        match self
            .request(SupervisorCommand::RawOutputAttempt {
                handle: handle.clone(),
                ordinal,
            })
            .await?
        {
            SupervisorResponse::RawOutputAttempt { lookup }
                if lookup.handle == handle
                    && lookup.ordinal == ordinal
                    && lookup
                        .attempt
                        .as_ref()
                        .is_none_or(|attempt| u32::from(attempt.ordinal) == ordinal) =>
            {
                Ok(lookup)
            }
            response => unexpected(response),
        }
    }

    pub async fn raw_output_page(
        &self,
        handle: runmat_execution::JobHandle,
        attempt_id: runmat_execution::identity::AttemptId,
        cursor: super::RawProcessOutputCursor,
    ) -> NativeExecutionResult<Option<super::RawProcessOutputPage>> {
        self.connect_or_start().await?;
        match self
            .request(SupervisorCommand::RawOutputPage {
                handle,
                attempt_id,
                cursor,
            })
            .await?
        {
            SupervisorResponse::RawOutputPage {
                attempt_id: received,
                page,
            } if received == attempt_id => {
                let Some(page) = page else { return Ok(None) };
                if page.attempt_id != attempt_id
                    || page.stdout.as_slice().len() > crate::process_output::OUTPUT_PAGE_BYTES
                    || page.stderr.as_slice().len() > crate::process_output::OUTPUT_PAGE_BYTES
                    || cursor
                        .stdout
                        .checked_add(page.stdout.as_slice().len() as u64)
                        != Some(page.next.stdout)
                    || cursor
                        .stderr
                        .checked_add(page.stderr.as_slice().len() as u64)
                        != Some(page.next.stderr)
                {
                    return Err(NativeExecutionError::Protocol(
                        "raw output response differs from requested attempt, cursor or page bound"
                            .into(),
                    ));
                }
                Ok(Some(*page))
            }
            response => unexpected(response),
        }
    }

    pub async fn cancel(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        self.connect_or_start().await?;
        match self.request(SupervisorCommand::Cancel { job_id }).await? {
            SupervisorResponse::Cancelled { record } => Ok(record),
            response => unexpected(response),
        }
    }

    pub async fn cancel_handle(
        &self,
        handle: runmat_execution::JobHandle,
        reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<LocalJobRecord> {
        self.connect_or_start().await?;
        match self
            .request(SupervisorCommand::CancelHandle { handle, reason })
            .await?
        {
            SupervisorResponse::Cancelled { record } => Ok(record),
            response => unexpected(response),
        }
    }

    async fn request(
        &self,
        command: SupervisorCommand,
    ) -> NativeExecutionResult<SupervisorResponse> {
        self.config.validate()?;
        // Profile the caller's command before opening a channel. The complete
        // authenticated envelope is bounded again before allocating its buffer.
        runmat_canonical_codec::profile_json_bytes(
            &command,
            u64::from(SUPERVISOR_MAX_MESSAGE_BYTES),
        )?;
        let mut stream = self.connect().await?;
        let token = read_token(&self.config.paths.token)?;
        let request = SupervisorRequest {
            local_fea_organization: self.local_fea_organization,
            protocol_version: SUPERVISOR_PROTOCOL_VERSION,
            authentication_token: token,
            config_identity: self.config.identity()?,
            command,
        };
        let payload = runmat_canonical_codec::encode_json_bounded(
            &request,
            u64::from(SUPERVISOR_MAX_MESSAGE_BYTES),
        )?;
        let limits = FrameLimits {
            max_message_bytes: SUPERVISOR_MAX_MESSAGE_BYTES,
        };
        write_payload(&mut stream, &payload, limits).await?;
        let payload = read_payload(&mut stream, limits).await?;
        let response: SupervisorResponse = serde_json::from_slice(&payload)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        match response {
            SupervisorResponse::Error { code, message } => {
                Err(NativeExecutionError::SupervisorRejected { code, message })
            }
            response => Ok(response),
        }
    }

    #[cfg(unix)]
    async fn connect(&self) -> NativeExecutionResult<impl AsyncRead + AsyncWrite + Unpin> {
        tokio::net::UnixStream::connect(&self.config.paths.socket)
            .await
            .map_err(|source| NativeExecutionError::SupervisorTransport { source })
    }

    #[cfg(windows)]
    async fn connect(&self) -> NativeExecutionResult<impl AsyncRead + AsyncWrite + Unpin> {
        use tokio::net::windows::named_pipe::ClientOptions;

        ClientOptions::new()
            .open(&self.config.paths.pipe)
            .map_err(|source| NativeExecutionError::SupervisorTransport { source })
    }
}

fn submitted(
    response: SupervisorResponse,
) -> NativeExecutionResult<(LocalJobRecord, JobSubmissionDisposition)> {
    match response {
        SupervisorResponse::Submitted { record, duplicate } => Ok((
            record,
            if duplicate {
                JobSubmissionDisposition::Duplicate
            } else {
                JobSubmissionDisposition::Created
            },
        )),
        response => unexpected(response),
    }
}

fn unexpected<T>(response: SupervisorResponse) -> NativeExecutionResult<T> {
    Err(NativeExecutionError::Protocol(format!(
        "local supervisor returned an unexpected response: {response:?}"
    )))
}

// Only endpoint absence permits a new authority bootstrap. Authentication,
// framing, permission and profile failures must never create another provider.
fn endpoint_absent(error: &NativeExecutionError) -> bool {
    matches!(error, NativeExecutionError::SupervisorTransport { source }
        if matches!(source.kind(), std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused))
}

#[cfg(all(test, unix))]
mod tests;

fn readiness_timeout() -> NativeExecutionError {
    NativeExecutionError::SupervisorTransport {
        source: std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "local supervisor did not respond within its readiness budget",
        ),
    }
}

mod startup;
