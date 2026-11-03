use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_time::Instant;

use runmat_execution::identity::WorkerId;
use runmat_execution::state::PoolState;
use runmat_execution::{CancellationReason, ExecutionScopeId, PoolId};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, PoolSpec, WorkerSpec,
};
use runmat_test_runner::worker::{
    BackendCapabilities, BackendError, BackendErrorKind, BackendFuture, CancelRequest,
    ExecutionRequest, SpawnRequest, WorkerBackend, WorkerExecution, WorkerSessionId,
};

use crate::request;
use crate::{ExecutionBackendConfig, ExecutionWorkerSession};

pub struct ExecutionWorkerBackend<B> {
    inner: B,
    config: ExecutionBackendConfig,
    capabilities: BackendCapabilities,
    sequence: AtomicU64,
}

impl<B: WorkerBackend> ExecutionWorkerBackend<B> {
    pub fn new(inner: B, config: ExecutionBackendConfig) -> Result<Self, BackendError> {
        config.validate()?;
        let mut capabilities = inner.capabilities();
        capabilities.host.max_workers = capabilities.host.max_workers.min(config.max_workers);
        if capabilities.host.max_workers == 0 {
            return Err(rejected(
                "execution-backed test host exposes no usable worker capacity",
            ));
        }
        Ok(Self {
            inner,
            config,
            capabilities,
            sequence: AtomicU64::new(0),
        })
    }
}

impl<B: WorkerBackend> WorkerBackend for ExecutionWorkerBackend<B> {
    type Session = ExecutionWorkerSession<B::Session>;

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn spawn<'a>(&'a self, request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        Box::pin(async move {
            let revision = request.submission.plan.program_revision.clone();
            if revision != request.submission.snapshot.program_revision {
                return Err(rejected(
                    "execution-backed test plan and snapshot revisions differ",
                ));
            }
            let run_id = request.submission.plan.run_id.clone();
            let inner = self.inner.spawn(request).await?;
            let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
            let scope_id = ExecutionScopeId::derive(&[
                b"runmat-test-execution-scope-v1",
                run_id.as_str().as_bytes(),
                &sequence.to_be_bytes(),
            ]);
            let pool_id = PoolId::derive(&[scope_id.bytes(), b"pool"]);
            let worker_id = WorkerId::derive(&[scope_id.bytes().as_slice(), b"worker".as_slice()]);
            let mut driver = Driver::new(
                DriverConfig {
                    max_in_flight: 1,
                    ..DriverConfig::default()
                },
                sequence.saturating_add(1),
            )
            .map_err(runner_error)?;
            driver
                .handle(DriverCommand::RegisterScope {
                    scope_id,
                    parent: None,
                })
                .map_err(runner_error)?;
            driver
                .handle(DriverCommand::CreatePool(PoolSpec {
                    id: pool_id,
                    min_workers: 1,
                    max_workers: 1,
                    max_in_flight: 1,
                    resource_limit: self.config.worker_resources.clone(),
                }))
                .map_err(runner_error)?;
            driver
                .handle(DriverCommand::SetPoolState {
                    pool_id,
                    state: PoolState::Ready,
                })
                .map_err(runner_error)?;
            driver
                .handle(DriverCommand::RegisterWorker(WorkerSpec {
                    id: worker_id,
                    pool_id,
                    resources: self.config.worker_resources.clone(),
                    host: crate::capability::host_inventory(&revision),
                }))
                .map_err(runner_error)?;
            Ok(ExecutionWorkerSession {
                id: WorkerSessionId(format!("execution:{}:{sequence}", run_id.as_str())),
                inner,
                driver: Arc::new(Mutex::new(driver)),
                scope_id,
                pool_id,
                worker_id,
                revision,
                active: Arc::new(Mutex::new(None)),
            })
        })
    }

    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        Box::pin(async move {
            let task = request::task(session, &request, &self.config);
            let launch = {
                let actions = session
                    .driver
                    .lock()
                    .expect("test execution driver poisoned")
                    .handle(DriverCommand::Submit(Box::new(task)))
                    .map_err(runner_error)?;
                one_launch(actions)?
            };
            *session
                .active
                .lock()
                .expect("test attempt registry poisoned") = Some(launch.clone());
            let started = Instant::now();
            let assignment = runmat_execution::ProgramExecutionAssignment {
                scope_id: launch.scope_id,
                pool_id: launch.task.pool_id,
                task_id: launch.task_id,
                attempt_id: launch.id,
                worker_id: launch.worker_id,
                backend: runmat_execution::PoolBackend::LocalProcesses,
                resources: launch.resource_assignment.clone(),
            };
            let execution = self
                .inner
                .execute_admitted(
                    &session.inner,
                    request,
                    assignment,
                    launch.task.resources.clone(),
                )
                .await;
            let report = match &execution {
                Ok(_) => AttemptReport::Succeeded {
                    result: metered_success(&launch, elapsed_millis(started)),
                },
                Err(error) => AttemptReport::Failed {
                    kind: failure_kind(error.kind),
                    message: error.to_string(),
                },
            };
            let mut active = session
                .active
                .lock()
                .expect("test attempt registry poisoned");
            // Cancellation may already have committed the terminal scheduler
            // report while the underlying backend was unwinding. In that
            // case the test-owned completion still returns to the coordinator,
            // but the fenced attempt must not be reported a second time.
            if active.as_ref().is_some_and(|active| active.id == launch.id) {
                session
                    .driver
                    .lock()
                    .expect("test execution driver poisoned")
                    .handle(DriverCommand::BackendReport(BackendReport::for_request(
                        &launch, report,
                    )))
                    .map_err(runner_error)?;
                *active = None;
            }
            execution
        })
    }

    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        Box::pin(async move {
            if let Some(active) = session
                .active
                .lock()
                .expect("test attempt registry poisoned")
                .as_ref()
            {
                let mut driver = session
                    .driver
                    .lock()
                    .expect("test execution driver poisoned");
                // Cancel this attempt, not the reusable worker's scope. The
                // coordinator's grace deadline is not the driver's clock.
                let now_millis = driver.snapshot_ref().now_millis;
                driver
                    .handle(DriverCommand::CancelTask {
                        task_id: active.task_id,
                        reason: CancellationReason::User,
                        now_millis,
                    })
                    .map_err(runner_error)?;
            }
            let execution = self.inner.cancel(&session.inner, request).await?;
            if let Some(completed) = &execution {
                let active = session
                    .active
                    .lock()
                    .expect("test attempt registry poisoned");
                if active.as_ref().is_some_and(|active| {
                    active.task_id
                        != session
                            .task_id(completed.result.test_id.as_str(), completed.result.attempt)
                }) {
                    return Err(rejected(
                        "cancellation completion differs from the active test attempt",
                    ));
                }
                drop(active);
                // A returned completed attempt certifies cooperative teardown;
                // an absent response does not certify physical retirement.
                retire_active(session)?;
            }
            Ok(execution)
        })
    }

    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            let _ = session
                .driver
                .lock()
                .expect("test execution driver poisoned")
                .handle(DriverCommand::WorkerLost(session.worker_id))
                .map_err(runner_error)?;
            // Some in-process hosts can only request cancellation. Their
            // transport acknowledgment is not a retirement certificate.
            self.inner.terminate(&session.inner).await
        })
    }

    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            let _ = session
                .driver
                .lock()
                .expect("test execution driver poisoned")
                .handle(DriverCommand::DrainWorker(session.worker_id))
                .map_err(runner_error)?;
            self.inner.shutdown(&session.inner).await
        })
    }
}

fn retire_active<S>(session: &ExecutionWorkerSession<S>) -> Result<(), BackendError> {
    let mut active = session
        .active
        .lock()
        .expect("test attempt registry poisoned");
    if let Some(request) = active.as_ref() {
        session
            .driver
            .lock()
            .expect("test execution driver poisoned")
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                request,
                AttemptReport::Retired,
            )))
            .map_err(runner_error)?;
        *active = None;
    }
    Ok(())
}

fn metered_success(
    request: &runmat_execution_runner::AttemptRequest,
    wall_millis: u64,
) -> AttemptSuccess {
    AttemptSuccess::new(
        request,
        vec![runmat_execution::value::ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::Null,
        ))],
        Vec::new(),
        wall_millis,
        0,
        0,
    )
    .expect("bounded test-runner allocation accounting must converge")
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn one_launch(
    actions: Vec<DriverAction>,
) -> Result<runmat_execution_runner::AttemptRequest, BackendError> {
    let mut launches = actions.into_iter().filter_map(|action| match action {
        DriverAction::Launch(request) => Some(request),
        _ => None,
    });
    let launch = launches
        .next()
        .ok_or_else(|| rejected("execution scheduler did not launch the test attempt"))?;
    if launches.next().is_some() {
        return Err(rejected(
            "execution scheduler launched more than one test attempt",
        ));
    }
    Ok(launch)
}

fn failure_kind(kind: BackendErrorKind) -> AttemptFailureKind {
    match kind {
        BackendErrorKind::Crashed | BackendErrorKind::Unavailable | BackendErrorKind::Transport => {
            AttemptFailureKind::Infrastructure
        }
        BackendErrorKind::Rejected => AttemptFailureKind::Rejected,
        BackendErrorKind::MalformedProtocol => AttemptFailureKind::Execution,
    }
}

fn runner_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::new(BackendErrorKind::Unavailable, error.to_string())
}

fn rejected(message: impl Into<String>) -> BackendError {
    BackendError::new(BackendErrorKind::Rejected, message)
}
