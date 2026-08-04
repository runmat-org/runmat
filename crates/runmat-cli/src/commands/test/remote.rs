use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundleBuilder, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_package::FrozenProjectHandoff;
use runmat_test::protocol::{ProtocolHandshake, WorkerCapability};
use runmat_test_runner::host::{HostCapabilities, IsolationMode};
use runmat_test_runner::worker::{
    BackendCapabilities, BackendError, BackendErrorKind, BackendFuture, CancelRequest,
    ExecutionRequest, RunSubmission, SpawnRequest, WorkerBackend, WorkerExecution, WorkerSessionId,
};
use runmat_test_runner_execution::{decode_execution, TestAttemptWorkload};
use uuid::Uuid;

use crate::commands::job::submit::{PreparedRemoteExecution, RemoteSubmissionOptions};

pub(super) struct RemoteTestBackend {
    config: RemoteTestBackendConfig,
    capabilities: BackendCapabilities,
    sequence: AtomicU64,
}

#[derive(Clone)]
pub(super) struct RemoteTestBackendConfig {
    pub project: Option<Uuid>,
    pub cluster: String,
    pub queue: String,
    pub trust_identity: String,
    pub max_workers: usize,
    pub project_handoff: FrozenProjectHandoff,
}

impl RemoteTestBackend {
    pub fn new(config: RemoteTestBackendConfig) -> Result<Self, BackendError> {
        if config.cluster.trim().is_empty()
            || config.queue.trim().is_empty()
            || config.trust_identity.trim().is_empty()
            || config.max_workers == 0
        {
            return Err(rejected(
                "remote tests require a cluster, queue, pinned trust identity, and non-zero worker bound",
            ));
        }
        config
            .project_handoff
            .validate()
            .map_err(|error| rejected(error.to_string()))?;
        Ok(Self {
            capabilities: BackendCapabilities {
                host: HostCapabilities::new([IsolationMode::Process], config.max_workers)
                    .map_err(|error| rejected(error.to_string()))?,
                handshake: ProtocolHandshake::current(
                    "runmat-remote-test-coordinator",
                    vec![
                        WorkerCapability::StrongIsolation,
                        WorkerCapability::CapturedOutput,
                        WorkerCapability::Artifacts,
                        WorkerCapability::Coverage,
                    ],
                ),
            },
            config,
            sequence: AtomicU64::new(0),
        })
    }
}

#[derive(Clone)]
pub(super) struct RemoteTestSession {
    id: WorkerSessionId,
    submission: RunSubmission,
    submission_key: Uuid,
    active_run: Arc<Mutex<Option<String>>>,
}

impl fmt::Debug for RemoteTestSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteTestSession")
            .field("id", &self.id)
            .field("run_id", &self.submission.plan.run_id)
            .finish_non_exhaustive()
    }
}

impl PartialEq for RemoteTestSession {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RemoteTestSession {}

impl WorkerBackend for RemoteTestBackend {
    type Session = RemoteTestSession;

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn spawn<'a>(&'a self, request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        Box::pin(async move {
            if request.isolation != IsolationMode::Process {
                return Err(rejected(
                    "remote test workers provide process isolation only",
                ));
            }
            let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
            Ok(RemoteTestSession {
                id: WorkerSessionId(format!(
                    "remote:{}:{sequence}",
                    request.submission.plan.run_id.as_str()
                )),
                submission: request.submission,
                submission_key: Uuid::new_v4(),
                active_run: Arc::new(Mutex::new(None)),
            })
        })
    }

    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        Box::pin(async move {
            let workload = TestAttemptWorkload::new(
                session.submission.clone(),
                request.test_id.clone(),
                request.attempt,
            )
            .map_err(protocol)?;
            let program = workload.program_request().map_err(protocol)?;
            let bundle = ExecutionBundleBuilder::native(
                &self.config.project_handoff.project,
                program.recipe.program_revision.clone(),
            )
            .map_err(protocol)?
            .with_materialized_program(
                program.recipe.clone(),
                ExecutableForm::TestAttemptV1,
                program.artifact.executable_bytes.clone(),
            )
            .build()
            .map_err(protocol)?;
            let mut bundle_archive = Vec::new();
            write_bundle(&bundle, &mut bundle_archive, ArchiveLimits::default())
                .map_err(protocol)?;
            let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                recipe: program.recipe.clone(),
                artifact: program.artifact.clone(),
                function: 0,
                requested_outputs: 1,
            })
            .map_err(protocol)?;
            let inputs = serde_json::to_vec(&ProgramExecutionInputs {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                arguments: Vec::new(),
            })
            .map_err(protocol)?;
            let active_run = Arc::clone(&session.active_run);
            let submitted = crate::commands::job::submit::submit_prepared(
                PreparedRemoteExecution {
                    revision: program.recipe.program_revision,
                    bundle_archive,
                    descriptor,
                    inputs,
                },
                RemoteSubmissionOptions {
                    project: self.config.project,
                    cluster: self.config.cluster.clone(),
                    queue: self.config.queue.clone(),
                    trust_identity: self.config.trust_identity.clone(),
                    idempotency_key: Some(format!(
                        "runmat-test-v1:{}:{}:{}",
                        session.submission_key,
                        request.test_id.as_str(),
                        request.attempt
                    )),
                    workers: 1,
                    on_run_created: Some(Arc::new(move |run_id| {
                        *active_run.lock().expect("remote test session poisoned") = Some(run_id);
                    })),
                },
            )
            .await
            .map_err(transport);
            let committed = match submitted {
                Ok(committed) => committed,
                Err(error) => {
                    *session
                        .active_run
                        .lock()
                        .expect("remote test session poisoned") = None;
                    return Err(error);
                }
            };
            *session
                .active_run
                .lock()
                .expect("remote test session poisoned") = Some(committed.id.clone());
            let response = crate::commands::job::attach::await_result(&committed.id)
                .await
                .map_err(transport);
            *session
                .active_run
                .lock()
                .expect("remote test session poisoned") = None;
            match response? {
                ProgramExecutionResponse::Success { value } => {
                    decode_execution(&value).map_err(protocol)
                }
                ProgramExecutionResponse::Failure { message } => {
                    Err(BackendError::new(BackendErrorKind::Crashed, message))
                }
            }
        })
    }

    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        _request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        Box::pin(async move {
            let run_id = session
                .active_run
                .lock()
                .expect("remote test session poisoned")
                .clone();
            if let Some(run_id) = run_id {
                crate::commands::job::cancel_remote_run(self.config.project, &run_id)
                    .await
                    .map_err(transport)?;
            }
            Ok(None)
        })
    }

    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            self.cancel(
                session,
                CancelRequest {
                    run_id: session.submission.plan.run_id.clone(),
                    reason: "worker terminated".into(),
                    grace_deadline_ms: 0,
                },
            )
            .await
            .map(|_| ())
        })
    }

    fn shutdown<'a>(&'a self, _session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

fn rejected(message: impl Into<String>) -> BackendError {
    BackendError::new(BackendErrorKind::Rejected, message)
}

fn protocol(error: impl std::fmt::Display) -> BackendError {
    BackendError::new(BackendErrorKind::MalformedProtocol, error.to_string())
}

fn transport(error: impl std::fmt::Display) -> BackendError {
    BackendError::new(BackendErrorKind::Transport, error.to_string())
}
