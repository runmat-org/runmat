use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
            let execution = self.inner.execute(&session.inner, request).await;
            let report = match &execution {
                Ok(_) => AttemptReport::Succeeded {
                    result: AttemptSuccess {
                        outputs: vec![runmat_execution::value::ValuePayload::Inline(Box::new(
                            runmat_execution::value::InlineValue::Null,
                        ))],
                        result_objects: Vec::new(),
                    },
                },
                Err(error) => AttemptReport::Failed {
                    kind: failure_kind(error.kind),
                    message: error.to_string(),
                },
            };
            let active = session
                .active
                .lock()
                .expect("test attempt registry poisoned")
                .take();
            // Cancellation may already have committed the terminal scheduler
            // report while the underlying backend was unwinding. In that
            // case the test-owned completion still returns to the coordinator,
            // but the fenced attempt must not be reported a second time.
            if active.is_some() {
                session
                    .driver
                    .lock()
                    .expect("test execution driver poisoned")
                    .handle(DriverCommand::BackendReport(BackendReport::for_request(
                        &launch, report,
                    )))
                    .map_err(runner_error)?;
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
            session
                .driver
                .lock()
                .expect("test execution driver poisoned")
                .handle(DriverCommand::CancelScope {
                    scope_id: session.scope_id,
                    reason: CancellationReason::User,
                    now_millis: request.grace_deadline_ms,
                })
                .map_err(runner_error)?;
            let execution = self.inner.cancel(&session.inner, request).await?;
            if let Some(active) = session
                .active
                .lock()
                .expect("test attempt registry poisoned")
                .take()
            {
                let report = if execution.is_some() {
                    AttemptReport::Succeeded {
                        result: AttemptSuccess {
                            outputs: vec![runmat_execution::value::ValuePayload::Inline(Box::new(
                                runmat_execution::value::InlineValue::Null,
                            ))],
                            result_objects: Vec::new(),
                        },
                    }
                } else {
                    AttemptReport::Cancelled
                };
                session
                    .driver
                    .lock()
                    .expect("test execution driver poisoned")
                    .handle(DriverCommand::BackendReport(BackendReport::for_request(
                        &active, report,
                    )))
                    .map_err(runner_error)?;
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
