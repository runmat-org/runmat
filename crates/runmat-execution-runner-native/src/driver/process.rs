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
    ChildOutput, ChildProcess, HostCommand, ProcessExit, ProcessHostError, ProcessHostResult,
    ProcessRetirementLease, RetirementBoundChild,
};
use tokio::io::BufReader;
pub(crate) mod graph;
pub(crate) mod owned;

use super::{
    LocalDriver, TaskCompletion, TransferFailure, TransferResult, NATIVE_ARTIFACT_STORE_ROOT_ENV,
    NATIVE_GRAPH_ATTEMPT_SCRATCH_ENV,
};
use crate::process_provider::{
    NATIVE_MESHING_SUBTASK_CACHE_ROOT_ENV, NATIVE_OBJECT_STORE_ROOT_ENV,
};
use crate::protocol::{
    CollectiveProcessResult, ReservedProgramExecutionRequest, StagedGraphWorkerLaunch,
    StagedGraphWorkerReference, StoredProgram, WorkerDriverMessage, WorkerLaunchRequest,
    WorkerProcessMessage, WorkerRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

/// Original evidence for one launched (or preflight-rejected) attempt.
pub(crate) struct LocalAttemptObservation {
    pub result: TransferResult,
    pub execution: AttemptExecutionObservation,
    pub unsuccessful_resources: Option<Box<AttemptResourceAccounting>>,
    pub console: Option<runmat_execution::console::ConsoleTranscript>,
    pub console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
    retirement: LocalRetirementEvidence,
}

/// Private physical-owner evidence, distinct from a public root-exit observation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum LocalRetirementEvidence {
    #[default]
    NotStarted,
    Unconfirmed,
    FullTreeRetired,
}

impl LocalAttemptObservation {
    pub(super) fn has_retirement_proof(&self) -> bool {
        !matches!(&self.result, Err(TransferFailure::RetirementUnverified(_)))
            && matches!(
                self.retirement,
                LocalRetirementEvidence::NotStarted | LocalRetirementEvidence::FullTreeRetired
            )
    }

    /// A cancellation result carries retirement proof only after the actual
    /// process owner returned. Other original errors are reported separately.
    pub(super) fn report(&self, verified_cancel: bool) -> AttemptReport {
        if !self.has_retirement_proof() {
            return AttemptReport::RetirementUnverified {
                message: self.result.as_ref().err().map_or_else(
                    || "local attempt has no original process-tree retirement proof".into(),
                    ToString::to_string,
                ),
            };
        }
        if verified_cancel {
            return AttemptReport::Retired;
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

    pub(crate) fn into_backend_report(
        self,
        request: &AttemptRequest,
        verified_cancel: bool,
    ) -> (TransferResult, runmat_execution_runner::port::BackendReport) {
        let mut report = runmat_execution_runner::port::BackendReport::for_request(
            request,
            self.report(verified_cancel),
        );
        report.execution = Some(Box::new(self.execution));
        report.unsuccessful_resources = self.unsuccessful_resources;
        report.console = self.console.map(Box::new);
        report.console_scopes = self.console_scopes;
        (self.result, report)
    }
}

#[derive(Default)]
struct TransportMeters {
    sent: IpcByteCounter,
    received: IpcByteCounter,
}

/// Evidence accumulated by the original process owner until its final observation.
struct ProcessAttemptEvidence {
    started: Instant,
    meters: TransportMeters,
    termination: AttemptExecutionTermination,
    console: Option<runmat_execution::console::ConsoleTranscript>,
    console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
    retirement: LocalRetirementEvidence,
}

impl ProcessAttemptEvidence {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            meters: TransportMeters::default(),
            termination: AttemptExecutionTermination::NotStarted,
            console: None,
            console_scopes: Vec::new(),
            retirement: LocalRetirementEvidence::NotStarted,
        }
    }
}

pub(crate) enum GraphProcessQuarantine {
    KnownChild {
        child: Box<RetirementBoundChild<GraphScratchLease>>,
        scratch_path: PathBuf,
        reason: String,
        retired: bool,
        /// A preserved RetirementBoundChild has already surrendered numeric
        /// group-signal authority; retain it only for durable quarantine.
        retryable: bool,
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

pub(crate) struct GraphScratchLease(PrivateArtifactLease);

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
    fn captured_stderr(&self) -> runmat_process_host::child::CapturedProcessOutput {
        match self {
            Self::Direct(child) => child.captured_stderr(),
            Self::Graph(child) => child.captured_stderr(),
        }
    }

    fn take_application_output(&mut self) -> ProcessHostResult<ChildOutput> {
        match self {
            Self::Direct(child) => child.take_application_output(),
            Self::Graph(child) => child.take_application_output(),
        }
    }

    async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        match self {
            Self::Direct(child) => child.wait().await,
            Self::Graph(child) => child.wait().await,
        }
    }

    /// Observe an owned leader before reaping it, then retire its entire
    /// process group while that PID is still pinned. Cancellation leaves the
    /// ProcessHost phase ready for a signal-free retirement retry.
    async fn wait_and_retire_tree(&mut self) -> ProcessHostResult<ProcessExit> {
        match self {
            Self::Direct(child) => child.wait_and_retire_tree().await,
            Self::Graph(child) => child.wait_and_retire_tree().await,
        }
    }

    async fn terminate_tree(&mut self) -> ProcessHostResult<()> {
        match self {
            Self::Direct(child) => child.terminate_tree().await,
            Self::Graph(child) => child.terminate_tree().await,
        }
    }

    async fn terminate_and_observe(&mut self) -> ProcessHostResult<ProcessExit> {
        self.terminate_tree().await?;
        self.wait().await
    }
}

/// The driver owns and joins this final retry. A panic or runtime failure
/// leaves the entry unverified and still owned for explicit persistence.
pub(crate) fn retry_unverified_quarantine(entries: &mut Vec<GraphProcessQuarantine>) {
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
                    if let GraphProcessQuarantine::KnownChild {
                        child,
                        retired,
                        retryable,
                        ..
                    } = entry
                    {
                        if *retryable {
                            *retired = runtime.block_on(child.terminate_tree()).is_ok();
                        }
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
pub(crate) fn persist_unverified_quarantine(entries: &mut Vec<GraphProcessQuarantine>) -> bool {
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
    let mut evidence = ProcessAttemptEvidence::new();
    let mut result = launch_attempt(driver, request, completion, &mut evidence);
    if request.task.console.is_some() {
        if let Some(owner) = driver.console_admission_owner() {
            if let Err(message) = owner.settle_subtree(
                driver.console_identity(request),
                evidence.console.as_ref(),
                &evidence.console_scopes,
            ) {
                result = Err(TransferFailure::Infrastructure(message));
            }
        }
    }
    let result = if let Some(message) = completion.control_failure() {
        match result {
            Err(TransferFailure::RetirementUnverified(message)) => {
                Err(TransferFailure::RetirementUnverified(message))
            }
            _ => Err(TransferFailure::Infrastructure(message)),
        }
    } else {
        result
    };
    evidence.finish(request, result)
}

fn launch_attempt(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
    evidence: &mut ProcessAttemptEvidence,
) -> Result<WorkerResponse, TransferFailure> {
    request
        .validate()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    // This attempt has not touched artifacts or launched a child. Preserve
    // its original cancellation instead of replacing it with staging failure.
    if completion.is_cancelled() {
        return Err(TransferFailure::Cancelled);
    }
    let console_budget = driver.console_subgrant(request)?;
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
        staged_graph_launch(request, graph, assignment, console_budget)?
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
        let reserved = Box::new(
            ReservedProgramExecutionRequest::new(program, request.task.resources.clone())
                .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
        );
        match console_budget {
            Some(console_budget) => WorkerLaunchRequest::ReservedConsole {
                request: reserved,
                console_budget,
            },
            None => WorkerLaunchRequest::Reserved(reserved),
        }
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
        WorkerLaunchRequest::Reserved(request)
        | WorkerLaunchRequest::ReservedFeaProxy { request, .. }
        | WorkerLaunchRequest::ReservedConsole { request, .. } => &request.request().context,
        WorkerLaunchRequest::ReservedFea(launch) => &launch.request.request().context,
        WorkerLaunchRequest::StagedGraph(request) => &request.staged_graph_v10.context,
    };
    let spmd_rank = match context {
        runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
            Some((task.gang.clone(), task.rank))
        }
        _ => None,
    };
    let owner = owned::ProcessOwner {
        attempt: request,
        config: &driver.config,
        artifacts: &driver.artifacts,
        objects: &driver.objects,
        graph_quarantine: &driver.graph_quarantine,
    };
    let control = owned::LocalControl { driver, completion };
    let control_owner = completion.raw_control_owner();
    let process = runmat_runtime::context::host_allocation::HostControlFuture::new(
        control_owner.as_ref(),
        || owned::run_process(&owner, worker_request, &control, evidence),
    )
    .map_err(control_allocation_failure)?;
    let result = runtime.block_on(process);
    if let Some((gang, rank)) = spmd_rank {
        driver.terminate_collective_rank(&gang, rank);
    }
    result
}

/// Allocation admission happens before launch and cannot be an output-journal failure.
fn control_allocation_failure(
    source: runmat_runtime::context::host_allocation::HostAllocationError,
) -> TransferFailure {
    let error = runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        None,
        "RunMat:Execution:CpuControl",
        format!("owned CPU control allocation: {source}"),
        source,
    );
    if runmat_runtime::runtime_error::error_has_control_cancellation_origin(&error) {
        return TransferFailure::Cancelled;
    }
    match runmat_runtime::execution::encode_runtime_failure(&error) {
        Ok(failure) => TransferFailure::Runtime(Box::new(failure)),
        Err(message) => TransferFailure::Infrastructure(message),
    }
}

fn staged_graph_launch(
    request: &AttemptRequest,
    graph: &super::GraphAttemptAuthority,
    assignment: runmat_execution::ProgramExecutionAssignment,
    console_budget: Option<runmat_execution::console::ConsoleBudget>,
) -> Result<WorkerLaunchRequest, TransferFailure> {
    if request.task.program_artifact_id() != Some(graph.request_artifact_id)
        || request.task.invocation_context != graph.context
        || request.task.outputs.requested_outputs != graph.requested_outputs
        || request.task.callable.program != graph.callable
        || request.task.environment != graph.invocation_controls.environment
        || request.task.console != graph.invocation_controls.console
        || !request.task.inputs.is_empty()
    {
        return Err(TransferFailure::Infrastructure(
            "Graph attempt differs from its admitted local submission".into(),
        ));
    }
    Ok(WorkerLaunchRequest::StagedGraph(Box::new(
        StagedGraphWorkerLaunch {
            staged_graph_v10: StagedGraphWorkerReference {
                fea_parent_admission: false,
                artifact_id: graph.artifact_id,
                stored_digest: graph.stored_digest,
                stored_length: graph.stored_length,
                assignment,
                context: graph.context.clone(),
                requested_outputs: graph.requested_outputs,
                resources: request.task.resources.clone(),
                console_budget,
                fea_grant: None,
            },
        },
    )))
}

impl ProcessAttemptEvidence {
    fn finish(
        self,
        request: &AttemptRequest,
        response: Result<WorkerResponse, TransferFailure>,
    ) -> LocalAttemptObservation {
        let Self {
            started,
            meters,
            termination,
            console,
            console_scopes,
            retirement,
        } = self;
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
                AttemptSuccess::new(request, outputs, Vec::new(), wall_millis, 0, 0)
                    .map_err(accounting)
            }
            WorkerResponse::ExternalizedSuccess {
                outputs,
                result_objects,
            } => AttemptSuccess::new(request, outputs, result_objects, wall_millis, 0, 0)
                .map_err(accounting),
            WorkerResponse::GraphSuccess { graph } => {
                AttemptSuccess::new_graph(request, graph, wall_millis, 0, 0).map_err(accounting)
            }
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
            console_scopes,
            retirement,
        }
    }
}

fn worker_exchange_failure(error: &ProcessHostError) -> TransferFailure {
    let worker_was_lost = matches!(error, ProcessHostError::Io(_));
    let message = error.to_string();
    if worker_was_lost {
        TransferFailure::WorkerLost(message)
    } else {
        TransferFailure::Infrastructure(message)
    }
}

/// Keep the original transport cause alive until its sibling application drain
/// and physical owner have finished. The primary message remains available to
/// existing cleanup diagnostics; it contains no premature stderr snapshot.
#[derive(Default)]
struct DeferredWorkerExchangeFailure(Option<ProcessHostError>);

impl DeferredWorkerExchangeFailure {
    fn record(&mut self, error: ProcessHostError) -> TransferFailure {
        let failure = worker_exchange_failure(&error);
        self.0 = Some(error);
        failure
    }

    fn clear(&mut self) {
        self.0 = None;
    }

    fn finish<T>(
        self,
        mut outcome: Result<T, TransferFailure>,
        stderr: &runmat_process_host::child::CapturedProcessOutput,
    ) -> Result<T, TransferFailure> {
        let Some(_original_transport) = self.0 else {
            return outcome;
        };
        let message = match &mut outcome {
            Err(
                TransferFailure::WorkerLost(message)
                | TransferFailure::Infrastructure(message)
                | TransferFailure::RetirementUnverified(message),
            ) => message,
            // A different primary failure keeps its existing precedence and
            // must not acquire diagnostics attributed to the transport cause.
            _ => return outcome,
        };
        use std::fmt::Write as _;
        match stderr.take_closed_snapshot() {
            Ok(snapshot) if !snapshot.bytes.is_empty() => {
                // Move the original bounded prefix out of its capture. Display
                // borrows those bytes; no text()/snapshot() payload clone or
                // second journal reader is introduced here.
                let _ = write!(
                    message,
                    "; worker stderr (observed_bytes={:?}): {snapshot}",
                    snapshot.observed_bytes
                );
            }
            Ok(_) => {}
            Err(error) => {
                let _ = write!(message, "; worker stderr capture unavailable: {error}");
            }
        }
        outcome
    }
}

#[cfg(test)]
mod tests;
