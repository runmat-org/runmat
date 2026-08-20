use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use runmat_process_host::{ChildProcess, HostCommand, ProcessHostError};
use runmat_test::event::TestEvent;
use runmat_test::protocol::{
    ProtocolHandshake, ProtocolLimits, WorkerCapability, WorkerRequest, WorkerResponse,
};
use runmat_test_runner::host::{HostCapabilities, IsolationMode};
use runmat_test_runner::worker::{
    validate_handshake, BackendCapabilities, BackendError, BackendErrorKind, BackendFuture,
    CancelRequest, ExecutionRequest, SpawnRequest, WorkerBackend, WorkerExecution, WorkerSessionId,
};
use tokio::io::BufReader;
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use crate::transport::{read_response, write_bootstrap, write_request, NativeWorkerBootstrap};
use crate::NativeRunnerError;

use super::command::ProcessBackendConfig;

pub struct ProcessBackend {
    config: ProcessBackendConfig,
    capabilities: BackendCapabilities,
    sequence: AtomicU64,
}

impl ProcessBackend {
    pub fn new(config: ProcessBackendConfig) -> Result<Self, BackendError> {
        super::pool::validate_capacity(config.max_workers)
            .map_err(|message| BackendError::new(BackendErrorKind::Rejected, message))?;
        if config.max_stderr_bytes == 0 {
            return Err(BackendError::new(
                BackendErrorKind::Rejected,
                "worker stderr bound must be greater than zero",
            ));
        }
        let handshake = ProtocolHandshake::current(
            "runmat-native-coordinator",
            vec![
                WorkerCapability::StrongIsolation,
                WorkerCapability::CapturedOutput,
                WorkerCapability::Artifacts,
                WorkerCapability::Coverage,
            ],
        );
        let host = HostCapabilities::new([IsolationMode::Process], config.max_workers)
            .map_err(|error| BackendError::new(BackendErrorKind::Rejected, error.to_string()))?;
        Ok(Self {
            config,
            capabilities: BackendCapabilities { host, handshake },
            sequence: AtomicU64::new(0),
        })
    }
}

#[derive(Clone)]
pub struct ProcessSession {
    id: WorkerSessionId,
    state: Arc<ProcessState>,
}

struct ProcessState {
    child: Mutex<ChildProcess>,
    process_id: Option<u32>,
    writer: Mutex<ChildStdin>,
    reader: Mutex<BufReader<ChildStdout>>,
    stderr: runmat_process_host::child::CapturedStderr,
    limits: ProtocolLimits,
}

impl fmt::Debug for ProcessSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessSession")
            .field("id", &self.id)
            .field("process_id", &self.state.process_id)
            .finish()
    }
}

impl PartialEq for ProcessSession {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ProcessSession {}

impl ProcessSession {
    pub fn id(&self) -> &WorkerSessionId {
        &self.id
    }

    pub fn captured_stderr(&self) -> String {
        self.state.stderr.text()
    }
}

impl WorkerBackend for ProcessBackend {
    type Session = ProcessSession;

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn spawn<'a>(&'a self, request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        Box::pin(async move {
            if request.isolation != IsolationMode::Process {
                return Err(BackendError::new(
                    BackendErrorKind::Rejected,
                    format!(
                        "native process backend cannot provide '{}' isolation",
                        request.isolation.as_str()
                    ),
                ));
            }
            let mut command = HostCommand::new(&self.config.executable);
            command.arguments = self.config.worker_arguments.clone();
            command.environment = self.config.environment.clone();
            command.environment_policy = self.config.environment_policy.clone();
            command.max_stderr_bytes = self.config.max_stderr_bytes;
            let mut child = command.spawn().await.map_err(spawn_error)?;
            let process_id = child.id();
            let stdio = child.take_stdio().map_err(spawn_error)?;
            let mut writer = stdio.stdin;
            let stdout = stdio.stdout;
            let stderr = child.captured_stderr();
            let mut reader = BufReader::new(stdout);
            let local = &self.capabilities.handshake;
            let initial_limits = local.limits;
            if let Err(error) = write_request(
                &mut writer,
                &WorkerRequest::Handshake(local.clone()),
                initial_limits,
            )
            .await
            {
                return Err(failed_spawn(&mut child, &stderr, native_backend_error(error)).await);
            }
            let remote = match read_response(&mut reader, initial_limits).await {
                Ok(WorkerResponse::Handshake(remote)) => remote,
                Ok(response) => {
                    return Err(failed_spawn(
                        &mut child,
                        &stderr,
                        BackendError::new(
                            BackendErrorKind::MalformedProtocol,
                            format!("worker returned {response:?} before handshake"),
                        ),
                    )
                    .await);
                }
                Err(error) => {
                    return Err(
                        failed_spawn(&mut child, &stderr, native_backend_error(error)).await,
                    );
                }
            };
            let limits = match validate_handshake(local, &remote) {
                Ok(limits) => limits,
                Err(error) => {
                    return Err(failed_spawn(
                        &mut child,
                        &stderr,
                        BackendError::new(BackendErrorKind::MalformedProtocol, error.to_string()),
                    )
                    .await);
                }
            };
            if let Err(error) = write_bootstrap(
                &mut writer,
                &NativeWorkerBootstrap::new(self.config.project_handoff.clone()),
                limits,
            )
            .await
            {
                return Err(failed_spawn(&mut child, &stderr, native_backend_error(error)).await);
            }
            let run_id = request.submission.plan.run_id.clone();
            let install = WorkerRequest::InstallPlan {
                plan: Box::new(request.submission.plan),
                snapshot: Box::new(request.submission.snapshot),
            };
            if let Err(error) = write_request(&mut writer, &install, limits).await {
                return Err(failed_spawn(&mut child, &stderr, native_backend_error(error)).await);
            }
            match read_response(&mut reader, limits).await {
                Ok(WorkerResponse::Ready { run_id: ready }) if ready == run_id => {}
                Ok(WorkerResponse::Rejected { code, message }) => {
                    return Err(failed_spawn(
                        &mut child,
                        &stderr,
                        BackendError::new(BackendErrorKind::Rejected, format!("{code}: {message}")),
                    )
                    .await);
                }
                Ok(response) => {
                    return Err(failed_spawn(
                        &mut child,
                        &stderr,
                        BackendError::new(
                            BackendErrorKind::MalformedProtocol,
                            format!("worker returned invalid plan response {response:?}"),
                        ),
                    )
                    .await);
                }
                Err(error) => {
                    return Err(
                        failed_spawn(&mut child, &stderr, native_backend_error(error)).await,
                    );
                }
            }
            let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
            Ok(ProcessSession {
                id: WorkerSessionId(format!(
                    "process:{}:{sequence}",
                    process_id.unwrap_or_default()
                )),
                state: Arc::new(ProcessState {
                    child: Mutex::new(child),
                    process_id,
                    writer: Mutex::new(writer),
                    reader: Mutex::new(reader),
                    stderr,
                    limits,
                }),
            })
        })
    }

    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        Box::pin(async move {
            send(
                session,
                &WorkerRequest::Execute {
                    test_id: request.test_id.clone(),
                    attempt: request.attempt,
                },
            )
            .await?;
            read_execution(session, &request.test_id, request.attempt).await
        })
    }

    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        Box::pin(async move {
            send(
                session,
                &WorkerRequest::Cancel {
                    run_id: request.run_id,
                    reason: request.reason,
                },
            )
            .await?;
            let mut reader = session.state.reader.lock().await;
            loop {
                match read_response(&mut *reader, session.state.limits)
                    .await
                    .map_err(native_backend_error)?
                {
                    WorkerResponse::Event { .. } => {}
                    WorkerResponse::Completed { .. } => {
                        // A completion can race with the cancellation frame after the worker has
                        // already left its active-test receive loop. In that case the unread frame
                        // would be interpreted as the next session request. A process-isolated
                        // session is therefore retired after every cancellation request; the
                        // coordinator records the requested terminal disposition and replaces the
                        // worker before executing more tests in the fixture group.
                        drop(reader);
                        let mut child = session.state.child.lock().await;
                        let _ = child.terminate_tree().await;
                        return Ok(None);
                    }
                    WorkerResponse::Rejected { .. } => return Ok(None),
                    response => {
                        return Err(BackendError::new(
                            BackendErrorKind::MalformedProtocol,
                            format!("unexpected cancellation response {response:?}"),
                        ));
                    }
                }
            }
        })
    }

    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            let mut child = session.state.child.lock().await;
            child.terminate_tree().await.map_err(transport_error)
        })
    }

    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            send(session, &WorkerRequest::Shutdown).await?;
            let mut reader = session.state.reader.lock().await;
            match read_response(&mut *reader, session.state.limits)
                .await
                .map_err(native_backend_error)?
            {
                WorkerResponse::ShutdownComplete => {
                    drop(reader);
                    let mut child = session.state.child.lock().await;
                    child.wait().await.map(|_| ()).map_err(transport_error)
                }
                response => Err(BackendError::new(
                    BackendErrorKind::MalformedProtocol,
                    format!("unexpected shutdown response {response:?}"),
                )),
            }
        })
    }
}

async fn send(session: &ProcessSession, request: &WorkerRequest) -> Result<(), BackendError> {
    let mut writer = session.state.writer.lock().await;
    write_request(&mut *writer, request, session.state.limits)
        .await
        .map_err(native_backend_error)
}

async fn read_execution(
    session: &ProcessSession,
    expected_test: &runmat_test::identity::TestId,
    expected_attempt: u32,
) -> Result<WorkerExecution, BackendError> {
    let mut reader = session.state.reader.lock().await;
    let mut events: Vec<TestEvent> = Vec::new();
    loop {
        match read_response(&mut *reader, session.state.limits)
            .await
            .map_err(|error| with_stderr(native_backend_error(error), session))?
        {
            WorkerResponse::Event { event } => events.push(event),
            WorkerResponse::Completed { result, coverage }
                if result.test_id == *expected_test && result.attempt == expected_attempt =>
            {
                return Ok(WorkerExecution {
                    result,
                    events,
                    coverage,
                });
            }
            WorkerResponse::Completed { result, .. } => {
                return Err(BackendError::new(
                    BackendErrorKind::MalformedProtocol,
                    format!(
                        "worker completed test '{}' attempt {} while '{}' attempt {} was active",
                        result.test_id.as_str(),
                        result.attempt,
                        expected_test.as_str(),
                        expected_attempt
                    ),
                ));
            }
            WorkerResponse::Rejected { code, message } => {
                return Err(BackendError::new(
                    BackendErrorKind::Rejected,
                    format!("{code}: {message}"),
                ));
            }
            response => {
                return Err(BackendError::new(
                    BackendErrorKind::MalformedProtocol,
                    format!("unexpected execution response {response:?}"),
                ));
            }
        }
    }
}

async fn failed_spawn(
    child: &mut ChildProcess,
    stderr: &runmat_process_host::child::CapturedStderr,
    error: BackendError,
) -> BackendError {
    let _ = child.terminate_tree().await;
    with_captured_stderr(error, stderr)
}

fn spawn_error(error: ProcessHostError) -> BackendError {
    BackendError::new(BackendErrorKind::Unavailable, error.to_string())
}

fn transport_error(error: ProcessHostError) -> BackendError {
    BackendError::new(BackendErrorKind::Transport, error.to_string())
}

fn native_backend_error(error: NativeRunnerError) -> BackendError {
    let kind = match &error {
        NativeRunnerError::Protocol(message) if message.contains("closed") => {
            BackendErrorKind::Crashed
        }
        NativeRunnerError::Protocol(_) => BackendErrorKind::MalformedProtocol,
        NativeRunnerError::Io(_) => BackendErrorKind::Transport,
        NativeRunnerError::Configuration(_) => BackendErrorKind::Rejected,
    };
    BackendError::new(kind, error.to_string())
}

fn with_stderr(mut error: BackendError, session: &ProcessSession) -> BackendError {
    append_stderr(&mut error, &session.captured_stderr());
    error
}

fn with_captured_stderr(
    mut error: BackendError,
    stderr: &runmat_process_host::child::CapturedStderr,
) -> BackendError {
    append_stderr(&mut error, &stderr.text());
    error
}

fn append_stderr(error: &mut BackendError, stderr: &str) {
    if !stderr.is_empty() {
        error.message.push_str("\nworker stderr:\n");
        error.message.push_str(stderr);
    }
}
