use runmat_execution_runner_native::fea_proxy;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use runmat_process_host::ipc::private_control::{ControlReader, ControlWriter};
use runmat_process_host::{ChildProcess, ProcessHostError};
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
use tokio::sync::Mutex;

use crate::transport::{read_response, write_bootstrap, write_request, NativeWorkerBootstrap};
use crate::NativeRunnerError;

use super::command::ProcessBackendConfig;

mod control;
mod output;
mod admission;

pub struct ProcessBackend {
    config: ProcessBackendConfig,
    capabilities: BackendCapabilities,
    sequence: AtomicU64,
    fea_admission: fea_proxy::Provider,
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
            fea_admission: fea_proxy::capture(),
        })
    }

    /// Keep the originating host in the parent; workers receive no credentials.
    pub fn with_fea_run_admission(
        mut self,
        admission: Arc<dyn runmat_runtime::analysis::fea_run_admission::FeaRunAdmission>,
    ) -> Self {
        self.fea_admission = Ok(admission);
        self
    }
}

#[derive(Clone)]
pub struct ProcessSession {
    id: WorkerSessionId,
    state: Arc<ProcessState>,
}

struct ProcessState {
    child: Mutex<runmat_process_host::child::RetirementBoundChild<admission::TestAdmissionLease>>,
    process_id: Option<u32>,
    writer: Mutex<ControlWriter>,
    reader: Mutex<BufReader<ControlReader>>,
    stdout: runmat_process_host::CapturedProcessOutput,
    stderr: runmat_process_host::child::CapturedProcessOutput,
    retired: AtomicBool,
    output_taken: AtomicBool,
    limits: ProtocolLimits,
    fea_admission: admission::ActiveAdmission,
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
            let local = &self.capabilities.handshake;
            let initial_limits = local.limits;
            let control::WorkerControl {
                mut child,
                mut writer,
                mut reader,
                stdout,
                stderr,
            } = control::spawn(&self.config).await?;
            let process_id = child.id();
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
                &NativeWorkerBootstrap::new(
                    self.config.project_handoff.clone(),
                    self.config.execution.clone(),
                ),
                limits,
            )
            .await
            {
                return Err(failed_spawn(&mut child, &stderr, native_backend_error(error)).await);
            }
            match read_response(&mut reader, limits).await {
                Ok(WorkerResponse::BootstrapReady) => {}
                response => {
                    return Err(failed_spawn(
                        &mut child,
                        &stderr,
                        BackendError::new(
                            BackendErrorKind::MalformedProtocol,
                            format!("invalid bootstrap acknowledgement: {response:?}"),
                        ),
                    )
                    .await)
                }
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
            let admission_lease = admission::TestAdmissionLease::default();
            let fea_admission = admission_lease.active.clone();
            let child =
                match runmat_process_host::child::RetirementBoundChild::new(child, admission_lease)
                {
                    Ok(child) => child,
                    Err(unbound) => {
                        let (mut child, _) = *unbound;
                        return Err(failed_spawn(
                            &mut child,
                            &stderr,
                            BackendError::new(
                                BackendErrorKind::Rejected,
                                "test worker must own its process tree",
                            ),
                        )
                        .await);
                    }
                };
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
                    stdout,
                    stderr,
                    retired: AtomicBool::new(false),
                    output_taken: AtomicBool::new(false),
                    limits,
                    fea_admission,
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
                    admission: None,
                },
            )
            .await?;
            read_execution(session, &request.test_id, request.attempt).await
        })
    }

    fn execute_admitted<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
        assignment: runmat_execution::ProgramExecutionAssignment,
        resources: runmat_execution::resource::ResourceRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        Box::pin(async move {
            let parent = fea_proxy::Parent::new(
                self.fea_admission.clone(),
                assignment.clone(),
                resources.clone(),
                None,
                0,
            )
            .map_err(|message| BackendError::new(BackendErrorKind::Rejected, message))?;
            {
                let mut active = session
                    .state
                    .fea_admission
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if active.is_some() {
                    return Err(BackendError::new(
                        BackendErrorKind::Rejected,
                        "worker already has an admitted execution",
                    ));
                }
                *active = Some(parent);
            }
            send(
                session,
                &WorkerRequest::Execute {
                    test_id: request.test_id.clone(),
                    attempt: request.attempt,
                    admission: Some(runmat_test::protocol::WorkerExecutionAdmission {
                        assignment,
                        resources,
                    }),
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
                match admission::read_response(session, &mut reader).await? {
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
                        child.terminate_tree().await.map_err(transport_error)?;
                        session.state.retired.store(true, Ordering::Release);
                        admission::close(session);
                        return Ok(None);
                    }
                    WorkerResponse::Rejected { .. } => {
                        drop(reader);
                        session
                            .state
                            .child
                            .lock()
                            .await
                            .terminate_tree()
                            .await
                            .map_err(transport_error)?;
                        session.state.retired.store(true, Ordering::Release);
                        admission::close(session);
                        return Ok(None);
                    }
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
            child.terminate_tree().await.map_err(transport_error)?;
            session.state.retired.store(true, Ordering::Release);
            admission::close(session);
            Ok(())
        })
    }

    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            send(session, &WorkerRequest::Shutdown).await?;
            let mut reader = session.state.reader.lock().await;
            match admission::read_response(session, &mut reader).await? {
                WorkerResponse::ShutdownComplete => {
                    drop(reader);
                    let mut child = session.state.child.lock().await;
                    child
                        .wait_and_retire_tree()
                        .await
                        .map_err(transport_error)?;
                    session.state.retired.store(true, Ordering::Release);
                    admission::close(session);
                    Ok(())
                }
                response => Err(BackendError::new(
                    BackendErrorKind::MalformedProtocol,
                    format!("unexpected shutdown response {response:?}"),
                )),
            }
        })
    }

    fn take_process_output<'a>(
        &'a self,
        session: &'a Self::Session,
    ) -> BackendFuture<'a, Option<runmat_test::event::WorkerProcessOutput>> {
        Box::pin(output::take(session))
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
        match admission::read_response(session, &mut reader)
            .await
            .map_err(|error| with_stderr(error, session))?
        {
            WorkerResponse::Event { event } => events.push(event),
            WorkerResponse::Completed { result, coverage }
                if result.test_id == *expected_test && result.attempt == expected_attempt =>
            {
                admission::finish(session)?;
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
    stderr: &runmat_process_host::child::CapturedProcessOutput,
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
        NativeRunnerError::ControlClosed { .. } => BackendErrorKind::Crashed,
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
    stderr: &runmat_process_host::child::CapturedProcessOutput,
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

#[cfg(test)]
mod admission_tests {
    use super::*;
    struct Uncalled;
    impl runmat_runtime::analysis::fea_run_admission::FeaRunAdmission for Uncalled {
        fn admit(
            &self,
            _: &runmat_runtime::analysis::fea_run_admission::FeaRunRequest,
        ) -> Result<
            Box<dyn runmat_runtime::analysis::fea_run_admission::FeaRunPermit>,
            runmat_runtime::analysis::fea_run_admission::FeaRunDenied,
        > {
            panic!("constructing a plain test backend must not admit FEA");
        }
    }
    #[test]
    fn explicit_host_is_retained_without_starting_a_lease() {
        let provider: Arc<dyn runmat_runtime::analysis::fea_run_admission::FeaRunAdmission> =
            Arc::new(Uncalled);
        let backend = ProcessBackend::new(ProcessBackendConfig::same_binary("unused-test-worker"))
            .unwrap()
            .with_fea_run_admission(provider.clone());
        assert!(Arc::ptr_eq(
            backend.fea_admission.as_ref().unwrap(),
            &provider
        ));
    }
}
