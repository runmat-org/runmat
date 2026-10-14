use std::path::PathBuf;
use std::time::{Duration, Instant};

use runmat_execution_artifact::PrivateArtifactLease;
use runmat_execution_runner::{AttemptRequest, AttemptSuccess};
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_process_host::{
    ChildProcess, ChildStdio, HostCommand, ProcessExit, ProcessHostError, ProcessHostResult,
    ProcessRetirementLease, RetirementBoundChild,
};
use tokio::io::BufReader;

use super::{
    LocalDriver, TaskCompletion, TransferFailure, TransferResult, NATIVE_ARTIFACT_STORE_ROOT_ENV,
    NATIVE_GRAPH_ATTEMPT_SCRATCH_ENV, NATIVE_OBJECT_STORE_ROOT_ENV,
};
use crate::protocol::{
    CollectiveProcessResult, StagedGraphWorkerLaunch, StagedGraphWorkerReference, StoredProgram,
    WorkerDriverMessage, WorkerLaunchRequest, WorkerProcessMessage, WorkerRequest, WorkerResponse,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

pub(super) enum GraphProcessQuarantine {
    KnownChild {
        child: Box<RetirementBoundChild<GraphScratchLease>>,
        scratch_path: PathBuf,
        reason: String,
        retired: bool,
    },
    /// Windows may fail JobObject assignment after a runnable child spawned.
    /// No root-only kill can certify its descendants, so retain the private
    /// byte lease and taint this driver even though ProcessHost returned no
    /// child handle to retry.
    SpawnUnverified {
        scratch: GraphScratchLease,
        scratch_path: PathBuf,
        reason: String,
    },
}

pub(super) struct GraphScratchLease(PrivateArtifactLease);

impl ProcessRetirementLease for GraphScratchLease {
    fn persist_unverified(&mut self, process_id: Option<u32>, reason: &str) -> Result<(), String> {
        self.0.persist_unverified(process_id, reason)
    }
}

enum AttemptChild {
    Direct(ChildProcess),
    Graph(RetirementBoundChild<GraphScratchLease>),
}

impl AttemptChild {
    fn captured_stderr(&self) -> runmat_process_host::child::CapturedStderr {
        match self {
            Self::Direct(child) => child.captured_stderr(),
            Self::Graph(child) => child.captured_stderr(),
        }
    }

    fn take_stdio(&mut self) -> ProcessHostResult<ChildStdio> {
        match self {
            Self::Direct(child) => child.take_stdio(),
            Self::Graph(child) => child.take_stdio(),
        }
    }

    async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        match self {
            Self::Direct(child) => child.wait().await,
            Self::Graph(child) => child.wait().await,
        }
    }

    async fn terminate_tree(&mut self) -> ProcessHostResult<()> {
        match self {
            Self::Direct(child) => child.terminate_tree().await,
            Self::Graph(child) => child.terminate_tree().await,
        }
    }
}

/// The driver owns and joins this final retry. A panic or runtime failure
/// leaves the entry unverified and still owned for explicit persistence.
pub(super) fn retry_unverified_quarantine(entries: &mut Vec<GraphProcessQuarantine>) {
    if entries.is_empty() {
        return;
    }
    std::thread::scope(|scope| {
        let retry = std::thread::Builder::new()
            .name("runmat-graph-retirement".into())
            .spawn_scoped(scope, || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                for entry in entries.iter_mut() {
                    if let GraphProcessQuarantine::KnownChild { child, retired, .. } = entry {
                        *retired = runtime.block_on(child.terminate_tree()).is_ok();
                    }
                }
            });
        if let Ok(retry) = retry {
            let _ = retry.join();
        }
    });
    entries.retain(|entry| {
        !matches!(
            entry,
            GraphProcessQuarantine::KnownChild { retired: true, .. }
        )
    });
}

/// Final fail-closed state when bounded tree-retirement retries did not prove
/// cleanup. Each exact private directory is recorded for operator recovery;
/// the driver root must not be recursively removed in this state.
pub(super) fn persist_unverified_quarantine(entries: &mut Vec<GraphProcessQuarantine>) -> bool {
    if entries.is_empty() {
        return true;
    }
    for entry in entries.drain(..) {
        let (scratch_path, pid, recorded) = match entry {
            GraphProcessQuarantine::KnownChild {
                mut child,
                scratch_path,
                reason,
                ..
            } => {
                let pid = child.id();
                let recorded = child.persist_unverified(&reason);
                // Dropping the child closes Owned root/JobObject containment,
                // but that is not retrospectively verified retirement.
                drop(child);
                (scratch_path, pid, recorded)
            }
            GraphProcessQuarantine::SpawnUnverified {
                mut scratch,
                scratch_path,
                reason,
            } => {
                let recorded = scratch.persist_unverified(None, &reason);
                (scratch_path, None, recorded)
            }
        };
        if let Err(error) = recorded {
            eprintln!(
                "Graph retirement quarantine marker could not be written for {} (pid {pid:?}): {error}",
                scratch_path.display()
            );
        } else {
            eprintln!(
                "Graph process retirement remains unverified; private scratch is quarantined at {}",
                scratch_path.display()
            );
        }
    }
    false
}

pub(super) fn execute_attempt(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
) -> TransferResult {
    request
        .validate()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let assignment = runmat_execution::ProgramExecutionAssignment {
        scope_id: request.scope_id,
        pool_id: request.task.pool_id,
        task_id: request.task_id,
        attempt_id: request.id,
        worker_id: request.worker_id,
        backend: runmat_execution::PoolBackend::LocalProcesses,
        resources: request.resource_assignment.clone(),
    };
    let worker_request = if let Some(graph) = completion.graph_authority.as_ref() {
        if request.task.program_artifact_id != graph.artifact_id
            || request.task.invocation_context != graph.context
            || request.task.outputs.requested_outputs != graph.requested_outputs
            || request.task.callable.program != graph.callable
            || !request.task.inputs.is_empty()
        {
            return Err(TransferFailure::Infrastructure(
                "Graph attempt differs from its admitted local submission".into(),
            ));
        }
        WorkerLaunchRequest::StagedGraph(Box::new(StagedGraphWorkerLaunch {
            staged_graph_v10: StagedGraphWorkerReference {
                artifact_id: graph.artifact_id,
                stored_digest: graph.stored_digest,
                stored_length: graph.stored_length,
                assignment,
                context: graph.context.clone(),
                requested_outputs: graph.requested_outputs,
            },
        }))
    } else {
        let stored = driver
            .artifacts
            .get(request.task.program_artifact_id)
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        let stored: StoredProgram = serde_json::from_slice(&stored)
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        let StoredProgram::Direct(stored) = stored else {
            return Err(TransferFailure::Infrastructure(
                "Graph local artifact lacks its typed response authority".into(),
            ));
        };
        WorkerLaunchRequest::Direct(Box::new(WorkerRequest::Direct(
            runmat_execution_artifact::ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe: stored.recipe,
                artifact: stored.artifact,
                callable: request.task.callable.program.clone(),
                context: request.task.invocation_context.clone(),
                assignment: Some(assignment),
                job_id: None,
                arguments: request.task.inputs.clone(),
                requested_outputs: request.task.outputs.requested_outputs,
            },
        )))
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let context = match &worker_request {
        WorkerLaunchRequest::Direct(request) => match request.as_ref() {
            WorkerRequest::Direct(request) => &request.context,
            WorkerRequest::Graph(_) => {
                return Err(TransferFailure::Infrastructure(
                    "Graph worker request must use an owned staged reference".into(),
                ));
            }
        },
        WorkerLaunchRequest::StagedGraph(request) => &request.staged_graph_v10.context,
    };
    let spmd_rank = match context {
        runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
            Some((task.gang.clone(), task.rank))
        }
        _ => None,
    };
    let result = runtime.block_on(run_process(driver, worker_request, completion));
    if let Some((gang, rank)) = spmd_rank {
        driver.terminate_collective_rank(&gang, rank);
    }
    result
}

async fn run_process(
    driver: &LocalDriver,
    request: WorkerLaunchRequest,
    completion: &TaskCompletion,
) -> TransferResult {
    #[cfg(windows)]
    if matches!(&request, WorkerLaunchRequest::StagedGraph(_)) {
        let own_executable = std::env::current_exe()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        let builtin_worker = [runmat_process_host::HiddenMode::ExecutionWorker
            .marker()
            .to_string()];
        if driver.config.executable != own_executable
            || driver.config.worker_arguments.as_slice() != builtin_worker.as_slice()
        {
            return Err(TransferFailure::Infrastructure(
                "Windows Graph containment requires RunMat's request-gated execution worker".into(),
            ));
        }
    }
    if !driver
        .graph_quarantine
        .lock()
        .expect("Graph quarantine lock poisoned")
        .is_empty()
    {
        return Err(TransferFailure::RetirementUnverified(
            "local worker has an unverified Graph process tree and refuses new work".into(),
        ));
    }
    if let WorkerLaunchRequest::Direct(request) = &request {
        request
            .validate()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    }
    let payload = serde_json::to_vec(&request)
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    if payload.len() > driver.config.max_message_bytes as usize {
        return Err(TransferFailure::Infrastructure(
            "worker request exceeds the native message bound".into(),
        ));
    }
    if completion.is_cancelled() {
        return Err(TransferFailure::Cancelled);
    }
    let graph_scratch = if matches!(&request, WorkerLaunchRequest::StagedGraph(_)) {
        Some(
            PrivateArtifactLease::new_attempt_scratch_in(&driver.store_root)
                .map_err(TransferFailure::Infrastructure)?,
        )
    } else {
        None
    };
    let graph_scratch_path = graph_scratch
        .as_ref()
        .map(|scratch| scratch.path().to_path_buf());
    let mut command = HostCommand::new(&driver.config.executable);
    command.arguments = driver.config.worker_arguments.clone();
    command.environment_policy = EnvironmentPolicy::Inherit;
    if matches!(&request, WorkerLaunchRequest::StagedGraph(_)) {
        command.environment_policy =
            EnvironmentPolicy::InheritExcept(vec!["RUNMAT_EXECUTION_WORKER_REMOTE".into()]);
    }
    command.environment.insert(
        NATIVE_OBJECT_STORE_ROOT_ENV.into(),
        driver.objects.root().to_string_lossy().into_owned(),
    );
    if matches!(&request, WorkerLaunchRequest::StagedGraph(_)) {
        command.environment.insert(
            NATIVE_ARTIFACT_STORE_ROOT_ENV.into(),
            driver.artifacts.root().to_string_lossy().into_owned(),
        );
        command.environment.insert(
            NATIVE_GRAPH_ATTEMPT_SCRATCH_ENV.into(),
            graph_scratch
                .as_ref()
                .expect("staged Graph has private attempt scratch")
                .path()
                .to_string_lossy()
                .into_owned(),
        );
    }
    command.max_stderr_bytes = driver.config.max_stderr_bytes;
    let child = match command.spawn().await {
        Ok(child) => child,
        Err(error) if graph_scratch.is_some() => {
            let message =
                format!("Graph worker spawn or process-tree containment is unverified: {error}");
            driver
                .graph_quarantine
                .lock()
                .expect("Graph quarantine lock poisoned")
                .push(GraphProcessQuarantine::SpawnUnverified {
                    scratch: GraphScratchLease(graph_scratch.expect("checked Graph scratch")),
                    scratch_path: graph_scratch_path.expect("checked Graph scratch path"),
                    reason: message.clone(),
                });
            return Err(TransferFailure::RetirementUnverified(message));
        }
        Err(error) => return Err(TransferFailure::Infrastructure(error.to_string())),
    };
    let mut child = match graph_scratch {
        Some(scratch) => match RetirementBoundChild::new(child, GraphScratchLease(scratch)) {
            Ok(bound) => AttemptChild::Graph(bound),
            Err(rejected) => {
                let (mut child, mut lease) = *rejected;
                let _ = lease.persist_unverified(
                    child.id(),
                    "selected Graph worker joined another process group",
                );
                let _ = child.terminate_tree().await;
                return Err(TransferFailure::RetirementUnverified(
                    "selected Graph worker lacks independent process-tree containment".into(),
                ));
            }
        },
        None => AttemptChild::Direct(child),
    };
    let mut child_retired = false;
    let outcome = async {
    if completion.is_cancelled() {
        if !matches!(&request, WorkerLaunchRequest::StagedGraph(_)) {
            let _ = child.terminate_tree().await;
        }
        return Err(TransferFailure::Cancelled);
    }
    let stderr = child.captured_stderr();
    let stdio = child
        .take_stdio()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let mut reader = BufReader::new(stdio.stdout);
    let mut writer = stdio.stdin;
    let limits = FrameLimits {
        max_message_bytes: driver.config.max_message_bytes,
    };
    write_payload(&mut writer, &payload, limits)
        .await
        .map_err(|error| worker_exchange_failure(error, &stderr.text()))?;
    let mut last_progress_sequence = 0;
    let mut graph_cancel_sent_at = None;
    let response = loop {
        let payload = tokio::select! {
            response = read_payload(&mut reader, limits) => {
                response.map_err(|error| worker_exchange_failure(error, &stderr.text()))?
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if completion.is_cancelled() {
                    if let WorkerLaunchRequest::StagedGraph(staged) = &request {
                        if let Some(sent_at) = graph_cancel_sent_at {
                            if Instant::now().duration_since(sent_at) >= Duration::from_secs(5) {
                                return Err(TransferFailure::Cancelled);
                            }
                        } else {
                            let payload = serde_json::to_vec(&WorkerDriverMessage::CancelAttempt {
                                attempt_id: staged.staged_graph_v10.assignment.attempt_id,
                                reason: completion.cancellation_reason().unwrap_or(
                                    runmat_execution::CancellationReason::User,
                                ),
                            })
                            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                            if !matches!(
                                tokio::time::timeout(
                                    Duration::from_secs(1),
                                    write_payload(&mut writer, &payload, limits)
                                )
                                .await,
                                Ok(Ok(()))
                            ) {
                                return Err(TransferFailure::Cancelled);
                            }
                            graph_cancel_sent_at = Some(Instant::now());
                        }
                    } else {
                        let _ = child.terminate_tree().await;
                        return Err(TransferFailure::Cancelled);
                    }
                }
                continue;
            }
        };
        if let Ok(message) = serde_json::from_slice::<WorkerProcessMessage>(&payload) {
            match message {
                WorkerProcessMessage::Progress { progress } => {
                    progress
                        .validate()
                        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                    if progress.sequence <= last_progress_sequence {
                        return Err(TransferFailure::Infrastructure(
                            "native worker progress is not strictly monotone".into(),
                        ));
                    }
                    last_progress_sequence = progress.sequence;
                    completion.record_progress(progress);
                }
                WorkerProcessMessage::CollectiveRequest { request } => {
                    let context = request.context.clone();
                    let id = request.id;
                    let sequence = request.sequence;
                    let result = match driver.collectives.execute(request, completion) {
                        Ok(response) => CollectiveProcessResult::Completed { response },
                        Err(message) => CollectiveProcessResult::Failed { message },
                    };
                    let payload = serde_json::to_vec(&WorkerDriverMessage::CollectiveCompletion {
                        context,
                        id,
                        sequence,
                        result,
                    })
                    .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                    write_payload(&mut writer, &payload, limits)
                        .await
                        .map_err(|error| worker_exchange_failure(error, &stderr.text()))?;
                }
                WorkerProcessMessage::Completed { response } => break response,
                WorkerProcessMessage::InfrastructureFailure { message } => {
                    // A root exit does not retire foreign-adapter descendants;
                    // the outer Graph error path still closes the group.
                    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
                    return Err(TransferFailure::Infrastructure(message));
                }
                WorkerProcessMessage::RetirementUnverified { message } => {
                    return Err(TransferFailure::RetirementUnverified(message));
                }
            }
        } else {
            break serde_json::from_slice::<WorkerResponse>(&payload)
                .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        }
    };
    if matches!(&request, WorkerLaunchRequest::StagedGraph(_)) {
        // This is a one-shot Graph child. Its control read may already have
        // entered Tokio's blocking stdin path; close our only writer after
        // the terminal frame so that read observes EOF before child wait.
        // No cancellation or collective command is legal after completion.
        drop(writer);
        if let Ok(Ok(_)) = tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
            // A completed root is not proof that its foreign-adapter children
            // have retired. Close the entire process group before publishing
            // a Graph terminal response or releasing its private roots.
            child.terminate_tree().await.map_err(|error| {
                TransferFailure::RetirementUnverified(format!(
                    "Graph worker replied, but its process tree could not be retired: {error}"
                ))
            })?;
            child_retired = true;
        } else {
            return Err(if completion.is_cancelled() {
                TransferFailure::Cancelled
            } else {
                TransferFailure::WorkerLost("Graph worker did not retire after its response".into())
            });
        }
    } else {
        let _ = child.wait().await;
    }
    match &request {
        WorkerLaunchRequest::Direct(request) => match request.as_ref() {
            WorkerRequest::Direct(request) => response.validate_against(request),
            WorkerRequest::Graph(_) => {
            return Err(TransferFailure::Infrastructure(
                "Graph worker response has no staged authority".into(),
            ));
            }
        },
        WorkerLaunchRequest::StagedGraph(_) => completion
            .graph_authority
            .as_ref()
            .ok_or_else(|| {
                TransferFailure::Infrastructure("Graph response lacks its typed authority".into())
            })?
            .response
            .validate_response(&response),
    }
    .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    if graph_cancel_sent_at.is_some() || completion.is_cancelled() {
        return Err(TransferFailure::Cancelled);
    }
    match response {
        WorkerResponse::Success { outputs } => Ok(AttemptSuccess::Values {
            outputs,
            result_objects: Vec::new(),
        }),
        WorkerResponse::GraphSuccess { graph } => Ok(AttemptSuccess::Graph { graph }),
        WorkerResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => Ok(AttemptSuccess::Values {
            outputs,
            result_objects,
        }),
        WorkerResponse::SpmdSuccess { outputs } => Ok(AttemptSuccess::Spmd { outputs }),
        WorkerResponse::Failure { message } => Err(TransferFailure::Execution(message)),
        WorkerResponse::RuntimeFailure { failure } => {
            Err(TransferFailure::Runtime(Box::new(failure)))
        }
    }
    }
    .await;
    if matches!(&request, WorkerLaunchRequest::StagedGraph(_)) && !child_retired {
        let mut retirement_error = None;
        for _ in 0..2 {
            match child.terminate_tree().await {
                Ok(()) => {
                    retirement_error = None;
                    break;
                }
                Err(error) => retirement_error = Some(error),
            }
        }
        if let Some(error) = retirement_error {
            let message = format!(
                "Graph worker process tree could not be retired after bounded retries: {error}"
            );
            let AttemptChild::Graph(child) = child else {
                unreachable!("a staged Graph worker owns a retirement-bound child")
            };
            driver
                .graph_quarantine
                .lock()
                .expect("Graph quarantine lock poisoned")
                .push(GraphProcessQuarantine::KnownChild {
                    child: Box::new(child),
                    scratch_path: graph_scratch_path
                        .expect("staged Graph has private attempt scratch"),
                    reason: message.clone(),
                    retired: false,
                });
            return Err(TransferFailure::RetirementUnverified(message));
        }
    }
    outcome
}

fn worker_exchange_failure(error: ProcessHostError, stderr: &str) -> TransferFailure {
    let worker_was_lost = matches!(&error, ProcessHostError::Io(_));
    let message = if stderr.is_empty() {
        error.to_string()
    } else {
        format!("{error}; worker stderr: {stderr}")
    };
    if worker_was_lost {
        TransferFailure::WorkerLost(message)
    } else {
        TransferFailure::Infrastructure(message)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::time::Duration;

    use runmat_execution::resource::Capability;
    use runmat_execution::{
        Digest, OutputContract, ProgramCallable, ProgramFunctionId, ProgramInvocationContext,
        ProgramRevision,
    };
    use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

    use super::{
        run_process, worker_exchange_failure, TransferFailure, WorkerLaunchRequest, WorkerRequest,
    };
    use crate::driver::{LocalDriver, TaskCompletion};

    mod selection;

    fn shell_driver(script: &str) -> (tempfile::TempDir, Arc<LocalDriver>) {
        let temporary = tempfile::tempdir().expect("temporary execution root");
        let scope = crate::config::fresh_scope_id(b"process-outcome-test", 1);
        let driver = LocalDriver::new(
            crate::NativeExecutionConfig {
                executable: "/bin/sh".into(),
                worker_arguments: vec!["-c".into(), script.into()],
                max_workers: 1,
                max_message_bytes: 64 * 1024,
                max_object_bytes: 64 * 1024,
                max_stderr_bytes: 4096,
                store_root: temporary.path().join("session"),
                worker_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
                host_inventory: runmat_core::RunMatSession::with_options(false, false)
                    .unwrap()
                    .execution_host_inventory(
                        runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                    )
                    .unwrap(),
                accelerator_devices: Vec::new(),
            },
            scope,
        )
        .expect("local driver");
        (temporary, driver)
    }

    fn request() -> WorkerRequest {
        let revision = ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .expect("program revision");
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "0".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("process-outcome-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV2,
            b"not-executed".to_vec(),
        )
        .expect("program artifact");
        WorkerRequest::Direct(runmat_execution_artifact::ProgramExecutionRequest {
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
            context: ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
        })
    }

    #[tokio::test]
    async fn dispatched_worker_exit_is_typed_as_worker_loss() {
        let (_temporary, driver) = shell_driver("dd bs=1 count=1 >/dev/null 2>&1; exit 42");
        let completion = TaskCompletion::new();
        let failure = run_process(
            &driver,
            WorkerLaunchRequest::Direct(Box::new(request())),
            &completion,
        )
        .await
        .expect_err("worker exit must fail");
        assert!(matches!(failure, TransferFailure::WorkerLost(_)));
    }

    #[tokio::test]
    async fn requested_process_termination_remains_typed_cancellation() {
        let (_temporary, driver) = shell_driver("dd bs=1 count=1 >/dev/null 2>&1; sleep 30");
        let completion = TaskCompletion::new();
        let execution = run_process(
            &driver,
            WorkerLaunchRequest::Direct(Box::new(request())),
            &completion,
        );
        tokio::pin!(execution);
        tokio::select! {
            result = &mut execution => panic!("worker exited before cancellation: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(50)) => completion.cancel(),
        }
        let failure = execution.await.expect_err("cancelled worker must fail");
        assert_eq!(failure, TransferFailure::Cancelled);
    }

    #[test]
    fn malformed_worker_frames_remain_infrastructure_failures() {
        let failure = worker_exchange_failure(
            runmat_process_host::ProcessHostError::Protocol("oversized frame".into()),
            "",
        );
        assert!(matches!(failure, TransferFailure::Infrastructure(_)));
    }
}
