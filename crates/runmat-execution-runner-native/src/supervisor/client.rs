use std::sync::Arc;
use std::time::Duration;

use runmat_execution::JobId;
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_process_host::{ChildLifetime, HiddenMode, HostCommand, StdioPolicy};
use tokio::io::{AsyncRead, AsyncWrite};

use super::auth::read_token;
use super::model::{BatchSubmission, JobAttachment, LocalJobRecord, ProgramBatchSubmission};
use super::protocol::{
    SupervisorCommand, SupervisorRequest, SupervisorResponse, SUPERVISOR_MAX_MESSAGE_BYTES,
    SUPERVISOR_PROTOCOL_VERSION,
};
use super::service::LocalSupervisorConfig;
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone)]
pub struct LocalSupervisorClient {
    config: Arc<LocalSupervisorConfig>,
}

impl LocalSupervisorClient {
    pub fn new(config: LocalSupervisorConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    pub fn for_current_executable() -> NativeExecutionResult<Self> {
        Ok(Self::new(LocalSupervisorConfig::for_current_executable()?))
    }

    pub async fn connect_or_start(&self) -> NativeExecutionResult<()> {
        if self.config.paths.token.exists() {
            read_token(&self.config.paths.token)?;
        }
        if self.ping().await.is_ok() {
            return Ok(());
        }
        let mut command = HostCommand::new(&self.config.executable);
        command.arguments = vec![HiddenMode::LocalSupervisor.marker().into()];
        command.environment_policy = EnvironmentPolicy::Inherit;
        command.lifetime = ChildLifetime::Detached;
        command.stdio = StdioPolicy::Null;
        command.max_stderr_bytes = self.config.max_stderr_bytes;
        let child = command.spawn().await?;
        drop(child);
        for _ in 0..100 {
            if self.ping().await.is_ok() {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Err(NativeExecutionError::Protocol(
            "the local execution supervisor did not become ready".into(),
        ))
    }

    pub async fn ping(&self) -> NativeExecutionResult<()> {
        match self.request(SupervisorCommand::Ping).await? {
            SupervisorResponse::Pong => Ok(()),
            response => unexpected(response),
        }
    }

    pub async fn submit(
        &self,
        submission: BatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        self.connect_or_start().await?;
        match self
            .request(SupervisorCommand::Submit {
                submission: Box::new(submission),
            })
            .await?
        {
            SupervisorResponse::Submitted { record, duplicate } => Ok((record, duplicate)),
            response => unexpected(response),
        }
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
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        self.connect_or_start().await?;
        match self
            .request(SupervisorCommand::SubmitProgram {
                submission: Box::new(submission),
            })
            .await?
        {
            SupervisorResponse::Submitted { record, duplicate } => Ok((record, duplicate)),
            response => unexpected(response),
        }
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
            SupervisorResponse::Attachment { attachment } => Ok(attachment),
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

    async fn request(
        &self,
        command: SupervisorCommand,
    ) -> NativeExecutionResult<SupervisorResponse> {
        let token = read_token(&self.config.paths.token)?;
        let request = SupervisorRequest {
            protocol_version: SUPERVISOR_PROTOCOL_VERSION,
            authentication_token: token,
            command,
        };
        let payload = serde_json::to_vec(&request)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let limits = FrameLimits {
            max_message_bytes: SUPERVISOR_MAX_MESSAGE_BYTES,
        };
        let mut stream = self.connect().await?;
        write_payload(&mut stream, &payload, limits).await?;
        let payload = read_payload(&mut stream, limits).await?;
        let response: SupervisorResponse = serde_json::from_slice(&payload)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        match response {
            SupervisorResponse::Error { code, message } => {
                Err(NativeExecutionError::Protocol(format!("{code}: {message}")))
            }
            response => Ok(response),
        }
    }

    #[cfg(unix)]
    async fn connect(&self) -> NativeExecutionResult<impl AsyncRead + AsyncWrite + Unpin> {
        tokio::net::UnixStream::connect(&self.config.paths.socket)
            .await
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
    }

    #[cfg(windows)]
    async fn connect(&self) -> NativeExecutionResult<impl AsyncRead + AsyncWrite + Unpin> {
        use tokio::net::windows::named_pipe::ClientOptions;

        ClientOptions::new()
            .open(&self.config.paths.pipe)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
    }
}

fn unexpected<T>(response: SupervisorResponse) -> NativeExecutionResult<T> {
    Err(NativeExecutionError::Protocol(format!(
        "local supervisor returned an unexpected response: {response:?}"
    )))
}
