use std::fs;
use std::sync::Arc;
use std::time::Duration;

use runmat_execution::JobId;
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_process_host::{
    is_process_alive, terminate_process_tree, ChildLifetime, HiddenMode, HostCommand, StdioPolicy,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::Mutex;

use super::auth::{constant_time_eq, load_or_create_token, SupervisorLock};
use super::filesystem::{secure_directory, unix_millis};
use super::model::{BatchSubmission, LocalJobRecord, LocalJobState, ProgramBatchSubmission};
use super::protocol::{
    SupervisorCommand, SupervisorRequest, SupervisorResponse, SUPERVISOR_MAX_MESSAGE_BYTES,
    SUPERVISOR_PROTOCOL_VERSION,
};
use super::store::{write_driver_marker, JobStore, SupervisorPaths};
use crate::{NativeExecutionError, NativeExecutionResult};

const STARTING_RECOVERY_GRACE_MILLIS: u64 = 10_000;

#[derive(Clone, Debug)]
pub struct LocalSupervisorConfig {
    pub executable: std::path::PathBuf,
    pub paths: SupervisorPaths,
    pub max_stderr_bytes: usize,
}

impl LocalSupervisorConfig {
    pub fn for_current_executable() -> NativeExecutionResult<Self> {
        Ok(Self {
            executable: std::env::current_exe().map_err(protocol_io)?,
            paths: SupervisorPaths::platform_default()?,
            max_stderr_bytes: 1024 * 1024,
        })
    }

    fn validate(&self) -> NativeExecutionResult<()> {
        if self.executable.as_os_str().is_empty() || self.max_stderr_bytes == 0 {
            return Err(NativeExecutionError::Configuration(
                "local supervisor configuration contains an empty bound".into(),
            ));
        }
        Ok(())
    }
}

pub struct LocalSupervisor {
    config: LocalSupervisorConfig,
    pub(super) store: Mutex<JobStore>,
}

impl LocalSupervisor {
    pub fn open(config: LocalSupervisorConfig) -> NativeExecutionResult<Arc<Self>> {
        config.validate()?;
        let store = JobStore::open(config.paths.clone())?;
        Ok(Arc::new(Self {
            config,
            store: Mutex::new(store),
        }))
    }

    pub async fn submit(
        &self,
        submission: BatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        let mut store = self.store.lock().await;
        let (record, created) = store.create_script(submission, unix_millis())?;
        if !created {
            return Ok((record, true));
        }
        let record = self.launch(&mut store, record).await?;
        Ok((record, false))
    }

    pub async fn submit_program(
        &self,
        submission: ProgramBatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        let mut store = self.store.lock().await;
        let (record, created) = store.create_program(submission, unix_millis())?;
        if !created {
            return Ok((record, true));
        }
        let record = self.launch(&mut store, record).await?;
        Ok((record, false))
    }

    pub async fn list(&self) -> NativeExecutionResult<Vec<LocalJobRecord>> {
        self.reconcile().await?;
        self.store.lock().await.list()
    }

    pub async fn show(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        self.reconcile().await?;
        self.store.lock().await.record(job_id)
    }

    pub async fn attach(
        &self,
        job_id: JobId,
        stdout_offset: u64,
        stderr_offset: u64,
    ) -> NativeExecutionResult<super::model::JobAttachment> {
        self.reconcile().await?;
        self.store
            .lock()
            .await
            .attachment(job_id, stdout_offset, stderr_offset)
    }

    pub async fn cancel(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        self.reconcile().await?;
        let store = self.store.lock().await;
        let mut record = store.record(job_id)?;
        if record.state.is_terminal() {
            return Ok(record);
        }
        let process_id = record.driver_process_id.or_else(|| {
            store
                .driver_marker(job_id)
                .ok()
                .flatten()
                .map(|value| value.process_id)
        });
        let now = unix_millis();
        if process_id.is_none() {
            record.state = LocalJobState::Cancelled;
            record.updated_unix_millis = now;
            record.message = Some("cancelled before driver start".into());
            store.write_record(&record)?;
            return Ok(record);
        }
        record.state = LocalJobState::Cancelling;
        record.driver_process_id = process_id;
        record.updated_unix_millis = now;
        store.write_record(&record)?;
        terminate_process_tree(process_id.expect("checked above"))
            .await
            .map_err(protocol_io)?;
        record.state = LocalJobState::Cancelled;
        record.driver_process_id = None;
        record.updated_unix_millis = unix_millis();
        record.message = Some("cancelled by user".into());
        store.write_record(&record)?;
        Ok(record)
    }

    pub async fn reconcile(&self) -> NativeExecutionResult<()> {
        let mut store = self.store.lock().await;
        let now = unix_millis();
        let records = store.list()?;
        for mut record in records {
            if record.state.is_terminal() {
                continue;
            }
            if let Some(completion) = store.completion(record.handle.id)? {
                record.state = if record.state == LocalJobState::Cancelling {
                    LocalJobState::Cancelled
                } else if completion.success {
                    LocalJobState::Succeeded
                } else {
                    LocalJobState::Failed
                };
                record.driver_process_id = None;
                record.exit_code = completion.exit_code;
                record.message = completion.message;
                record.updated_unix_millis = now;
                store.write_record(&record)?;
                continue;
            }
            match record.state {
                LocalJobState::Queued => {
                    let _ = self.launch(&mut store, record).await?;
                }
                LocalJobState::Starting => {
                    if let Some(marker) = store.driver_marker(record.handle.id)? {
                        if is_process_alive(marker.process_id) {
                            record.state = LocalJobState::Running;
                            record.driver_process_id = Some(marker.process_id);
                            record.updated_unix_millis = now;
                            store.write_record(&record)?;
                        } else {
                            mark_indeterminate(
                                &store,
                                &mut record,
                                now,
                                "driver exited during supervisor handoff",
                            )?;
                        }
                    } else if now.saturating_sub(record.updated_unix_millis)
                        >= STARTING_RECOVERY_GRACE_MILLIS
                    {
                        mark_indeterminate(
                            &store,
                            &mut record,
                            now,
                            "supervisor restarted during driver launch",
                        )?;
                    }
                }
                LocalJobState::Running | LocalJobState::Cancelling => {
                    let process_id = record.driver_process_id.expect("validated running record");
                    if !is_process_alive(process_id) {
                        if record.state == LocalJobState::Cancelling {
                            record.state = LocalJobState::Cancelled;
                            record.message = Some("driver stopped after cancellation".into());
                            record.driver_process_id = None;
                            record.updated_unix_millis = now;
                            store.write_record(&record)?;
                        } else {
                            mark_indeterminate(
                                &store,
                                &mut record,
                                now,
                                "driver exited without a committed completion",
                            )?;
                        }
                    }
                }
                LocalJobState::Succeeded
                | LocalJobState::Failed
                | LocalJobState::Cancelled
                | LocalJobState::Indeterminate => {}
            }
        }
        let _ = store.gc(now)?;
        Ok(())
    }

    async fn launch(
        &self,
        store: &mut JobStore,
        mut record: LocalJobRecord,
    ) -> NativeExecutionResult<LocalJobRecord> {
        record.state = LocalJobState::Starting;
        record.driver_process_id = None;
        record.updated_unix_millis = unix_millis();
        store.write_record(&record)?;

        let job_dir = store.job_dir(record.handle.id);
        let mut command = HostCommand::new(&self.config.executable);
        command.arguments = vec![HiddenMode::ExecutionDriver.marker().into()];
        command.environment_policy = EnvironmentPolicy::Inherit;
        command.environment.insert(
            "RUNMAT_EXECUTION_JOB_DIR".into(),
            job_dir.to_string_lossy().into_owned(),
        );
        command.lifetime = ChildLifetime::Detached;
        command.stdio = StdioPolicy::Files {
            stdout: store.stdout_path(record.handle.id),
            stderr: store.stderr_path(record.handle.id),
        };
        command.max_stderr_bytes = self.config.max_stderr_bytes;
        let child = match command.spawn().await {
            Ok(child) => child,
            Err(error) => {
                record.state = LocalJobState::Failed;
                record.updated_unix_millis = unix_millis();
                record.message = Some(format!("failed to launch local batch driver: {error}"));
                store.write_record(&record)?;
                return Ok(record);
            }
        };
        let process_id = child.id().ok_or_else(|| {
            NativeExecutionError::Protocol("batch driver did not expose a process id".into())
        })?;
        write_driver_marker(&job_dir, process_id)?;
        drop(child);
        record.state = LocalJobState::Running;
        record.driver_process_id = Some(process_id);
        record.updated_unix_millis = unix_millis();
        store.write_record(&record)?;
        Ok(record)
    }

    async fn command(
        &self,
        command: SupervisorCommand,
    ) -> NativeExecutionResult<SupervisorResponse> {
        Ok(match command {
            SupervisorCommand::Ping => SupervisorResponse::Pong,
            SupervisorCommand::Submit { submission } => {
                let (record, duplicate) = self.submit(*submission).await?;
                SupervisorResponse::Submitted { record, duplicate }
            }
            SupervisorCommand::SubmitProgram { submission } => {
                let (record, duplicate) = self.submit_program(*submission).await?;
                SupervisorResponse::Submitted { record, duplicate }
            }
            SupervisorCommand::List => SupervisorResponse::Jobs {
                records: self.list().await?,
            },
            SupervisorCommand::Show { job_id } => SupervisorResponse::Job {
                record: self.show(job_id).await?,
            },
            SupervisorCommand::Attach {
                job_id,
                stdout_offset,
                stderr_offset,
            } => SupervisorResponse::Attachment {
                attachment: self.attach(job_id, stdout_offset, stderr_offset).await?,
            },
            SupervisorCommand::Cancel { job_id } => SupervisorResponse::Cancelled {
                record: self.cancel(job_id).await?,
            },
        })
    }
}

pub async fn run_local_supervisor(config: LocalSupervisorConfig) -> NativeExecutionResult<()> {
    secure_directory(&config.paths.root)?;
    let _lock = SupervisorLock::acquire(&config.paths)?;
    let token = load_or_create_token(&config.paths)?;
    let supervisor = LocalSupervisor::open(config.clone())?;
    supervisor.reconcile().await?;
    let background = Arc::clone(&supervisor);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(250));
        loop {
            interval.tick().await;
            let _ = background.reconcile().await;
        }
    });
    run_listener(config.paths, supervisor, token).await
}

async fn handle_connection<S>(mut stream: S, supervisor: Arc<LocalSupervisor>, token: Arc<String>)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let limits = FrameLimits {
        max_message_bytes: SUPERVISOR_MAX_MESSAGE_BYTES,
    };
    let response = match read_payload(&mut stream, limits).await {
        Ok(payload) => match serde_json::from_slice::<SupervisorRequest>(&payload) {
            Ok(request)
                if request.protocol_version == SUPERVISOR_PROTOCOL_VERSION
                    && constant_time_eq(
                        request.authentication_token.as_bytes(),
                        token.as_bytes(),
                    ) =>
            {
                supervisor
                    .command(request.command)
                    .await
                    .unwrap_or_else(|error| {
                        SupervisorResponse::error("request_failed", error.to_string())
                    })
            }
            Ok(_) => SupervisorResponse::error(
                "authentication_failed",
                "local supervisor authentication or protocol negotiation failed",
            ),
            Err(error) => SupervisorResponse::error("malformed_request", error.to_string()),
        },
        Err(error) => SupervisorResponse::error("invalid_frame", error.to_string()),
    };
    if let Ok(payload) = serde_json::to_vec(&response) {
        let _ = write_payload(&mut stream, &payload, limits).await;
    }
}

#[cfg(unix)]
async fn run_listener(
    paths: SupervisorPaths,
    supervisor: Arc<LocalSupervisor>,
    token: String,
) -> NativeExecutionResult<()> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    if paths.socket.exists() {
        fs::remove_file(&paths.socket).map_err(protocol_io)?;
    }
    let listener = tokio::net::UnixListener::bind(&paths.socket).map_err(protocol_io)?;
    fs::set_permissions(&paths.socket, fs::Permissions::from_mode(0o600)).map_err(protocol_io)?;
    let token_owner = fs::metadata(&paths.token).map_err(protocol_io)?.uid();
    let token = Arc::new(token);
    loop {
        let (stream, _) = listener.accept().await.map_err(protocol_io)?;
        let peer_owner = stream.peer_cred().map_err(protocol_io)?.uid();
        if peer_owner != token_owner {
            continue;
        }
        tokio::spawn(handle_connection(
            stream,
            Arc::clone(&supervisor),
            Arc::clone(&token),
        ));
    }
}

#[cfg(windows)]
async fn run_listener(
    paths: SupervisorPaths,
    supervisor: Arc<LocalSupervisor>,
    token: String,
) -> NativeExecutionResult<()> {
    use tokio::net::windows::named_pipe::ServerOptions;

    let token = Arc::new(token);
    let mut first = true;
    loop {
        let server = ServerOptions::new()
            .first_pipe_instance(first)
            .reject_remote_clients(true)
            .create(&paths.pipe)
            .map_err(protocol_io)?;
        first = false;
        server.connect().await.map_err(protocol_io)?;
        tokio::spawn(handle_connection(
            server,
            Arc::clone(&supervisor),
            Arc::clone(&token),
        ));
    }
}

fn mark_indeterminate(
    store: &JobStore,
    record: &mut LocalJobRecord,
    now: u64,
    message: &str,
) -> NativeExecutionResult<()> {
    record.state = LocalJobState::Indeterminate;
    record.driver_process_id = None;
    record.updated_unix_millis = now;
    record.message = Some(message.into());
    store.write_record(record)
}

fn protocol_io(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
