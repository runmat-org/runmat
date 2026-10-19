//! Launch one admitted local attempt and retain its original provider evidence:
//! observed interval, process termination, IPC volume, console transcript and
//! reservation accounting travel with the scheduler report.
use std::path::PathBuf;
use std::time::{Duration, Instant};

use runmat_execution_artifact::PrivateArtifactLease;
use runmat_execution_runner::{
    AttemptExecutionObservation, AttemptExecutionTermination, AttemptFailureKind, AttemptReport,
    AttemptRequest, AttemptResourceAccounting, AttemptSuccess,
};
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{
    read_payload, write_payload, FrameLimits, IpcByteCounter, MeteredIo,
};
use runmat_process_host::{
    ChildProcess, ChildStdio, HostCommand, ProcessExit, ProcessHostError, ProcessHostResult,
    ProcessRetirementLease, RetirementBoundChild,
};
use tokio::io::BufReader;

use super::{
    LocalDriver, TaskCompletion, TransferFailure, TransferResult, NATIVE_ARTIFACT_STORE_ROOT_ENV,
    NATIVE_GRAPH_ATTEMPT_SCRATCH_ENV,
};
use crate::process_provider::{NATIVE_MESHING_SUBTASK_CACHE_ROOT_ENV, NATIVE_OBJECT_STORE_ROOT_ENV};
use crate::protocol::{
    CollectiveProcessResult, ReservedProgramExecutionRequest, StagedGraphWorkerLaunch,
    StagedGraphWorkerReference, StoredProgram, WorkerDriverMessage, WorkerLaunchRequest,
    WorkerProcessMessage, WorkerRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

/// Original evidence for one launched (or preflight-rejected) attempt.
pub(super) struct LocalAttemptObservation {
    pub result: TransferResult,
    pub execution: AttemptExecutionObservation,
    pub unsuccessful_resources: Option<Box<AttemptResourceAccounting>>,
    pub console: Option<runmat_execution::console::ConsoleTranscript>,
}

impl LocalAttemptObservation {
    /// Build the scheduler report. A verified Graph cancellation is reported
    /// as cleanup proof, never as an ordinary success or failure.
    pub(super) fn report(&self, verified_graph_cancel: bool) -> AttemptReport {
        if verified_graph_cancel {
            return AttemptReport::GraphRetiredCancelled;
        }
        match &self.result {
            Ok(success) => AttemptReport::Succeeded {
                result: success.clone(),
            },
            Err(TransferFailure::Execution(message)) => AttemptReport::Failed {
                kind: AttemptFailureKind::Execution,
                message: message.clone(),
            },
            Err(TransferFailure::Infrastructure(message)) => AttemptReport::Failed {
                kind: AttemptFailureKind::Infrastructure,
                message: message.clone(),
            },
            Err(TransferFailure::WorkerLost(message)) => AttemptReport::Lost {
                message: message.clone(),
            },
            Err(TransferFailure::RetirementUnverified(message)) => {
                AttemptReport::RetirementUnverified {
                    message: message.clone(),
                }
            }
            Err(TransferFailure::Cancelled) => AttemptReport::Cancelled,
            Err(TransferFailure::Runtime(failure)) => AttemptReport::RuntimeFailed {
                failure: *failure.clone(),
            },
        }
    }

    pub(super) fn into_backend_report(
        self,
        request: &AttemptRequest,
        verified_graph_cancel: bool,
    ) -> (TransferResult, runmat_execution_runner::port::BackendReport) {
        let mut report = runmat_execution_runner::port::BackendReport::for_request(
            request,
            self.report(verified_graph_cancel),
        );
        report.execution = Some(Box::new(self.execution));
        report.unsuccessful_resources = self.unsuccessful_resources;
        report.console = self.console.map(Box::new);
        (self.result, report)
    }
}

#[derive(Default)]
struct TransportMeters {
    sent: IpcByteCounter,
    received: IpcByteCounter,
}

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
) -> LocalAttemptObservation {
    let started = Instant::now();
    let meters = TransportMeters::default();
    let mut termination = AttemptExecutionTermination::NotStarted;
    let mut console = None;
    let result = launch_attempt(
        driver,
        request,
        completion,
        &meters,
        &mut termination,
        &mut console,
    );
    observe(request, started, result, meters, termination, console)
}

fn launch_attempt(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
    meters: &TransportMeters,
    termination: &mut AttemptExecutionTermination,
    console: &mut Option<runmat_execution::console::ConsoleTranscript>,
) -> Result<WorkerResponse, TransferFailure> {
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
        if request.task.program_artifact_id() != Some(graph.artifact_id)
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
        let artifact_id = request.task.program_artifact_id().ok_or_else(|| {
            TransferFailure::Infrastructure("native program attempt has no program artifact".into())
        })?;
        request
            .task
            .validate_invocation_reservation()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        if let Some(limits) = request.task.console {
            limits
                .validate_transport(u64::from(driver.config.max_message_bytes))
                .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        }
        let invocation = runmat_execution::console::invocation_memory_bytes(
            &request.task.environment,
            request.task.console,
        )
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        if invocation
            .checked_add(runmat_process_host::ipc::WORKER_PARENT_MONITOR_STACK_BYTES)
            .is_none_or(|minimum| minimum > request.task.resources.memory_bytes)
        {
            return Err(TransferFailure::Infrastructure(
                "native invocation and parent monitor exceed task memory reservation".into(),
            ));
        }
        let stored = driver
            .artifacts
            .get(artifact_id, u64::from(driver.config.max_message_bytes))
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        let stored: StoredProgram = serde_json::from_slice(&stored)
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        let StoredProgram::Direct(stored) = stored else {
            return Err(TransferFailure::Infrastructure(
                "Graph local artifact lacks its typed response authority".into(),
            ));
        };
        let program = runmat_execution_artifact::ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe: stored.recipe,
            artifact: stored.artifact,
            callable: request.task.callable.program.clone(),
            context: request.task.invocation_context.clone(),
            assignment: Some(assignment),
            job_id: None,
            arguments: request.task.inputs.clone(),
            requested_outputs: request.task.outputs.requested_outputs,
            environment: request.task.environment.clone(),
            console: request.task.console,
        };
        // Never repair a malformed stored entrypoint or invoke a foreign artifact.
        if runmat_execution::identity::ArtifactId::derive(&[program.artifact.id.0.bytes()])
            != artifact_id
        {
            return Err(TransferFailure::Infrastructure(
                "native stored program differs from original admitted artifact".into(),
            ));
        }
        // The worker receives its exact reservation and admits CPU parallelism within it.
        WorkerLaunchRequest::Reserved(Box::new(
            ReservedProgramExecutionRequest::new(program, request.task.resources.clone())
                .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
        ))
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
        WorkerLaunchRequest::Reserved(request) => &request.request().context,
        WorkerLaunchRequest::StagedGraph(request) => &request.staged_graph_v10.context,
    };
    let spmd_rank = match context {
        runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
            Some((task.gang.clone(), task.rank))
        }
        _ => None,
    };
    let result = runtime.block_on(run_process(
        driver,
        worker_request,
        completion,
        meters,
        termination,
        console,
    ));
    if let Some((gang, rank)) = spmd_rank {
        driver.terminate_collective_rank(&gang, rank);
    }
    result
}

async fn run_process(
    driver: &LocalDriver,
    request: WorkerLaunchRequest,
    completion: &TaskCompletion,
    meters: &TransportMeters,
    termination: &mut AttemptExecutionTermination,
    console: &mut Option<runmat_execution::console::ConsoleTranscript>,
) -> Result<WorkerResponse, TransferFailure> {
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
    match &request {
        WorkerLaunchRequest::Direct(request) => request
            .validate()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
        WorkerLaunchRequest::Reserved(request) => request
            .validate()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
        WorkerLaunchRequest::StagedGraph(_) => {}
    }
    let console_controls = match &request {
        WorkerLaunchRequest::Reserved(request) => request.request().console,
        WorkerLaunchRequest::Direct(_) | WorkerLaunchRequest::StagedGraph(_) => None,
    };
    let mut console_receiver = console_controls
        .map(runmat_execution::console::ConsoleReceiver::new)
        .transpose()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let mut console_receipt = None;
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
            PrivateArtifactLease::new_attempt_scratch_in(&driver.config.store_root)
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
    command.environment.insert(
        NATIVE_MESHING_SUBTASK_CACHE_ROOT_ENV.into(),
        driver
            .config
            .meshing_subtask_cache_root
            .clone()
            .unwrap_or_else(|| driver.config.store_root.join("meshing-subtask-cache/v1"))
            .to_string_lossy()
            .into_owned(),
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
    // Spawn can fail after creating a child (for example during containment).
    // Until root exit is observed, even an error leaves launch unconfirmed.
    *termination = AttemptExecutionTermination::Unconfirmed;
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
    let mut reader = BufReader::new(MeteredIo::new(stdio.stdout, meters.received.clone()));
    let mut writer = MeteredIo::new(stdio.stdin, meters.sent.clone());
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
                WorkerProcessMessage::Console { record } => {
                    console_receiver
                        .as_mut()
                        .ok_or_else(|| {
                            TransferFailure::Infrastructure(
                                "worker emitted unrequested console evidence".into(),
                            )
                        })?
                        .push(record)
                        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                }
                WorkerProcessMessage::Completed {
                    response,
                    console: receipt,
                } => {
                    match (console_receiver.as_ref(), receipt.as_ref()) {
                        (Some(receiver), Some(receipt)) => receiver
                            .verify_receipt(receipt)
                            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
                        (None, None) => {}
                        _ => {
                            return Err(TransferFailure::Infrastructure(
                                "worker console receipt differs from requested controls".into(),
                            ))
                        }
                    }
                    console_receipt = receipt;
                    break response;
                }
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
        if let Ok(Ok(exit)) = tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
            *termination = AttemptExecutionTermination::ProcessExited {
                code: exit.code,
                success: exit.success,
            };
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
        let exit = child.wait().await.map_err(|error| {
            TransferFailure::WorkerLost(format!("worker exit was not observed: {error}"))
        })?;
        *termination = AttemptExecutionTermination::ProcessExited {
            code: exit.code,
            success: exit.success,
        };
        if !exit.success {
            return Err(TransferFailure::Execution(format!(
                "worker returned a response but exited unsuccessfully: {:?}",
                exit.code
            )));
        }
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
        WorkerLaunchRequest::Reserved(request) => response.validate_against(request.request()),
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
    Ok(response)
    }
    .await;
    *console = console_receiver
        .map(|receiver| receiver.finish(console_receipt.as_ref()))
        .transpose()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
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

fn observe(
    request: &AttemptRequest,
    started: Instant,
    response: Result<WorkerResponse, TransferFailure>,
    meters: TransportMeters,
    termination: AttemptExecutionTermination,
    console: Option<runmat_execution::console::ConsoleTranscript>,
) -> LocalAttemptObservation {
    let wall_millis = u64::try_from(started.elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let execution = AttemptExecutionObservation {
        attempt_id: request.id,
        worker_id: request.worker_id,
        wall_millis,
        termination,
        local_ipc_sent_bytes: meters.sent.bytes(),
        local_ipc_received_bytes: meters.received.bytes(),
    };
    let accounting = |error: String| TransferFailure::Infrastructure(error);
    let mut result = response.and_then(|response| match response {
        WorkerResponse::Success { outputs } => {
            AttemptSuccess::new(request, outputs, Vec::new(), wall_millis, 0, 0).map_err(accounting)
        }
        WorkerResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => AttemptSuccess::new(request, outputs, result_objects, wall_millis, 0, 0)
            .map_err(accounting),
        WorkerResponse::GraphSuccess { graph } => Ok(AttemptSuccess::Graph { graph }),
        WorkerResponse::SpmdSuccess { outputs } => Ok(AttemptSuccess::Spmd { outputs }),
        WorkerResponse::Failure { message } => Err(TransferFailure::Execution(message)),
        WorkerResponse::RuntimeFailure { failure } => {
            Err(TransferFailure::Runtime(Box::new(failure)))
        }
    });
    // Preflight only reads the admitted program. No process or result producer
    // has been invoked, so output/egress/relay consumption is known zero. Once
    // launch is attempted, no such claim can be inferred from its eventual error.
    let unsuccessful_resources =
        if result.is_err() && termination == AttemptExecutionTermination::NotStarted {
            match AttemptResourceAccounting::for_attempt(request, wall_millis, 0, 0, 0) {
                Ok(resources) => Some(Box::new(resources)),
                Err(error) => {
                    result = Err(TransferFailure::Infrastructure(format!(
                        "{}; preflight accounting unavailable: {error}",
                        result.as_ref().expect_err("unsuccessful preflight")
                    )));
                    None
                }
            }
        } else {
            None
        };
    LocalAttemptObservation {
        result,
        execution,
        unsuccessful_resources,
        console,
    }
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

#[cfg(test)]
mod tests;
