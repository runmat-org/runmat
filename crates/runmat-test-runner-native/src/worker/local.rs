use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use runmat_core::RunMatSession;
use runmat_package::FrozenProjectHandoff;
use runmat_test::identity::{RunId, TestId};
use runmat_test_runner::host::{HostCapabilities, IsolationMode};
use runmat_test_runner::worker::{
    BackendCapabilities, BackendError, BackendErrorKind, BackendFuture, CancelRequest,
    ExecutionRequest, RunSubmission, SpawnRequest, WorkerBackend, WorkerExecution, WorkerSessionId,
};
use tokio::sync::watch;

#[derive(Clone, Debug)]
pub struct LocalBackendConfig {
    pub isolation: IsolationMode,
    pub enable_jit: bool,
    pub max_workers: usize,
    pub project_handoff: Option<FrozenProjectHandoff>,
}

impl LocalBackendConfig {
    pub fn new(isolation: IsolationMode) -> Self {
        Self {
            isolation,
            enable_jit: true,
            max_workers: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            project_handoff: None,
        }
    }
}

pub struct LocalBackend {
    config: LocalBackendConfig,
    capabilities: BackendCapabilities,
    shared_none: Mutex<Option<(RunId, Arc<LocalState>)>>,
    sequence: AtomicU64,
}

impl LocalBackend {
    pub fn new(config: LocalBackendConfig) -> Result<Self, BackendError> {
        if !matches!(
            config.isolation,
            IsolationMode::Session | IsolationMode::None
        ) {
            return Err(rejected(
                "local backend provides only explicit session or none isolation",
            ));
        }
        let max_workers = if config.isolation == IsolationMode::None {
            1
        } else {
            config.max_workers
        };
        let host = HostCapabilities::new([config.isolation], max_workers)
            .map_err(|error| rejected(error.to_string()))?;
        Ok(Self {
            config,
            capabilities: BackendCapabilities {
                host,
                handshake: runmat_test::protocol::ProtocolHandshake::current(
                    "runmat-native-local-coordinator",
                    Vec::new(),
                ),
            },
            shared_none: Mutex::new(None),
            sequence: AtomicU64::new(0),
        })
    }

    fn spawn_state(&self, submission: RunSubmission) -> Result<Arc<LocalState>, BackendError> {
        let (sender, receiver) = mpsc::channel();
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let enable_jit = self.config.enable_jit;
        let project = self.config.project_handoff.clone();
        std::thread::Builder::new()
            .name("runmat-test-session".into())
            .spawn(move || worker_loop(receiver, ready_sender, submission, enable_jit, project))
            .map_err(|error| {
                BackendError::new(
                    BackendErrorKind::Unavailable,
                    format!("failed to create local test session: {error}"),
                )
            })?;
        ready_receiver.recv().map_err(|_| {
            BackendError::new(
                BackendErrorKind::Crashed,
                "local test session exited during initialization",
            )
        })??;
        Ok(Arc::new(LocalState {
            sender,
            active: Mutex::new(None),
        }))
    }
}

#[derive(Clone)]
pub struct LocalSession {
    id: WorkerSessionId,
    mode: IsolationMode,
    run_id: RunId,
    state: Arc<LocalState>,
}

impl fmt::Debug for LocalSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalSession")
            .field("id", &self.id)
            .field("mode", &self.mode)
            .finish()
    }
}

impl PartialEq for LocalSession {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for LocalSession {}

struct LocalState {
    sender: mpsc::Sender<LocalCommand>,
    active: Mutex<Option<ActiveExecution>>,
}

#[derive(Clone)]
struct ActiveExecution {
    cancellation: Arc<AtomicBool>,
    result: watch::Receiver<Option<Result<WorkerExecution, BackendError>>>,
}

enum LocalCommand {
    Execute {
        test_id: TestId,
        attempt: u32,
        cancellation: Arc<AtomicBool>,
        result: watch::Sender<Option<Result<WorkerExecution, BackendError>>>,
    },
    Shutdown,
}

impl WorkerBackend for LocalBackend {
    type Session = LocalSession;

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn spawn<'a>(&'a self, request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        Box::pin(async move {
            if request.isolation != self.config.isolation {
                return Err(rejected(format!(
                    "local backend cannot provide '{}' isolation",
                    request.isolation.as_str()
                )));
            }
            let run_id = request.submission.plan.run_id.clone();
            let state = if self.config.isolation == IsolationMode::None {
                let mut shared = self
                    .shared_none
                    .lock()
                    .expect("local backend lock poisoned");
                if let Some((installed_run, state)) = shared.as_ref() {
                    if *installed_run != run_id {
                        return Err(rejected(
                            "none-isolated backend cannot install two runs concurrently",
                        ));
                    }
                    state.clone()
                } else {
                    let state = self.spawn_state(request.submission)?;
                    *shared = Some((run_id.clone(), state.clone()));
                    state
                }
            } else {
                self.spawn_state(request.submission)?
            };
            let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
            Ok(LocalSession {
                id: WorkerSessionId(format!("{}:{sequence}", self.config.isolation.as_str())),
                mode: self.config.isolation,
                run_id,
                state,
            })
        })
    }

    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        Box::pin(async move {
            let cancellation = Arc::new(AtomicBool::new(false));
            let (result_sender, result) = watch::channel(None);
            {
                let mut active = session
                    .state
                    .active
                    .lock()
                    .expect("local session lock poisoned");
                if active.is_some() {
                    return Err(rejected("local session is already executing a test"));
                }
                *active = Some(ActiveExecution {
                    cancellation: cancellation.clone(),
                    result: result.clone(),
                });
            }
            if session
                .state
                .sender
                .send(LocalCommand::Execute {
                    test_id: request.test_id,
                    attempt: request.attempt,
                    cancellation,
                    result: result_sender,
                })
                .is_err()
            {
                clear_active(&session.state);
                return Err(crashed("local test session has exited"));
            }
            let completed = await_result(result).await;
            clear_active(&session.state);
            completed
        })
    }

    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        Box::pin(async move {
            if request.run_id != session.run_id {
                return Err(rejected("cancellation targeted a different run"));
            }
            let active = session
                .state
                .active
                .lock()
                .expect("local session lock poisoned")
                .clone();
            let Some(active) = active else {
                return Ok(None);
            };
            active.cancellation.store(true, Ordering::Release);
            await_result(active.result).await.map(Some)
        })
    }

    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            if let Some(active) = session
                .state
                .active
                .lock()
                .expect("local session lock poisoned")
                .as_ref()
            {
                active.cancellation.store(true, Ordering::Release);
            }
            Ok(())
        })
    }

    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            if session.mode == IsolationMode::Session {
                let _ = session.state.sender.send(LocalCommand::Shutdown);
            }
            Ok(())
        })
    }
}

async fn await_result(
    mut result: watch::Receiver<Option<Result<WorkerExecution, BackendError>>>,
) -> Result<WorkerExecution, BackendError> {
    loop {
        if let Some(completed) = result.borrow().clone() {
            return completed;
        }
        result
            .changed()
            .await
            .map_err(|_| crashed("local test session exited without a result"))?;
    }
}

fn clear_active(state: &LocalState) {
    *state.active.lock().expect("local session lock poisoned") = None;
}

fn worker_loop(
    commands: mpsc::Receiver<LocalCommand>,
    ready: mpsc::SyncSender<Result<(), BackendError>>,
    submission: RunSubmission,
    enable_jit: bool,
    project: Option<FrozenProjectHandoff>,
) {
    let mut session = match RunMatSession::with_options(enable_jit, false) {
        Ok(session) => session,
        Err(error) => {
            let _ = ready.send(Err(crashed(format!(
                "failed to initialize local test session: {error}"
            ))));
            return;
        }
    };
    if let Some(project) = project {
        if let Err(error) = session.install_project_handoff(project) {
            let _ = ready.send(Err(rejected(format!(
                "failed to install frozen project in local session: {error}"
            ))));
            return;
        }
    }
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready.send(Err(crashed(format!(
                "failed to initialize local test runtime: {error}"
            ))));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Ok(command) = commands.recv() {
        match command {
            LocalCommand::Execute {
                test_id,
                attempt,
                cancellation,
                result,
            } => {
                let completed = runtime
                    .block_on(session.execute_planned_test(
                        &submission.snapshot,
                        &submission.plan,
                        &test_id,
                        attempt,
                        cancellation,
                    ))
                    .map(|attempt| WorkerExecution {
                        result: attempt.result,
                        events: attempt.events,
                    })
                    .map_err(|error| rejected(error.to_string()));
                let _ = result.send(Some(completed));
            }
            LocalCommand::Shutdown => return,
        }
    }
}

fn rejected(message: impl Into<String>) -> BackendError {
    BackendError::new(BackendErrorKind::Rejected, message)
}

fn crashed(message: impl Into<String>) -> BackendError {
    BackendError::new(BackendErrorKind::Crashed, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_strong_isolation_names() {
        for mode in [
            IsolationMode::Auto,
            IsolationMode::Process,
            IsolationMode::Worker,
        ] {
            let error = LocalBackend::new(LocalBackendConfig::new(mode))
                .err()
                .expect("mode must be rejected");
            assert_eq!(error.kind, BackendErrorKind::Rejected);
        }
    }

    #[test]
    fn none_is_explicitly_single_lane() {
        let backend =
            LocalBackend::new(LocalBackendConfig::new(IsolationMode::None)).expect("none backend");
        assert_eq!(backend.capabilities().host.max_workers, 1);
        assert!(backend.capabilities().host.supports(IsolationMode::None));
        assert!(!backend.capabilities().host.supports(IsolationMode::Process));
    }
}
