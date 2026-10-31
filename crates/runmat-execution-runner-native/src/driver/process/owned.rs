//! The existing owned process body, shared without duplicating Graph retirement.
use super::super::GraphAttemptAuthority;
use super::*;

pub(crate) struct ProcessOwner<'a> {
    pub(crate) config: &'a crate::NativeExecutionConfig,
    pub(crate) artifacts: &'a crate::local_store::ArtifactStore,
    pub(crate) objects: &'a crate::FilesystemObjectStore,
    pub(crate) graph_quarantine: &'a std::sync::Mutex<Vec<GraphProcessQuarantine>>,
}
pub(crate) trait OwnedProcessControl: Send + Sync {
    fn is_cancelled(&self) -> bool;
    fn cancellation_reason(&self) -> Option<runmat_execution::CancellationReason>;
    fn record_progress(&self, progress: crate::protocol::ProgramProgress);
    fn graph_authority(&self) -> Option<&GraphAttemptAuthority>;
    fn record_console(
        &self,
        _record: &runmat_execution::console::ConsoleRecord,
    ) -> Result<(), String> {
        Ok(())
    }
    fn finish_console(
        &self,
        _receipt: &runmat_execution::console::ConsoleReceipt,
    ) -> Result<(), String> {
        Ok(())
    }
    fn execute_collective(
        &self,
        _request: runmat_execution::CollectiveRequest,
    ) -> Result<runmat_execution::CollectiveResponse, String> {
        Err("Graph attempt has no admitted collective authority".into())
    }
}
pub(super) struct LocalControl<'a> {
    pub(super) driver: &'a LocalDriver,
    pub(super) completion: &'a TaskCompletion,
}
impl OwnedProcessControl for LocalControl<'_> {
    fn is_cancelled(&self) -> bool {
        self.completion.is_cancelled()
    }
    fn cancellation_reason(&self) -> Option<runmat_execution::CancellationReason> {
        self.completion.cancellation_reason()
    }
    fn record_progress(&self, progress: crate::protocol::ProgramProgress) {
        self.completion.record_progress(progress)
    }
    fn graph_authority(&self) -> Option<&GraphAttemptAuthority> {
        self.completion.graph_authority.as_ref()
    }
    fn execute_collective(
        &self,
        request: runmat_execution::CollectiveRequest,
    ) -> Result<runmat_execution::CollectiveResponse, String> {
        self.driver.collectives.execute(request, self.completion)
    }
}

pub(super) async fn run_process(
    driver: &ProcessOwner<'_>,
    request: WorkerLaunchRequest,
    completion: &dyn OwnedProcessControl,
    meters: &TransportMeters,
    termination: &mut AttemptExecutionTermination,
    console: &mut Option<runmat_execution::console::ConsoleTranscript>,
    retirement: &mut LocalRetirementEvidence,
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
        WorkerLaunchRequest::StagedGraph(_) => {
            completion
                .graph_authority()
                .ok_or_else(|| {
                    TransferFailure::Infrastructure(
                        "Graph console has no admitted authority".into(),
                    )
                })?
                .invocation_controls
                .console
        }
        WorkerLaunchRequest::Direct(_) => None,
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
    // Until full-tree retirement is observed, even an error leaves launch unconfirmed.
    *termination = AttemptExecutionTermination::Unconfirmed;
    *retirement = LocalRetirementEvidence::Unconfirmed;
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
    let mut child_preserved_reason = None;
    let outcome = async {
    if completion.is_cancelled() {
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
        let payload = {
        // Keep this same read alive across cancellation ticks: read_exact can
        // already have consumed part of the header or body before Pending.
        let read = read_payload(&mut reader, limits);
        tokio::pin!(read);
        loop { tokio::select! {
            response = &mut read => {
                break response.map_err(|error| worker_exchange_failure(error, &stderr.text()))?;
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
                                reason: completion.cancellation_reason().ok_or_else(||
                                    TransferFailure::Infrastructure("Graph stop has no authenticated cancellation reason".into()))?,
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
                        return Err(TransferFailure::Cancelled);
                    }
                }
                continue;
            }
        }}
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
                    let result = match completion.execute_collective(request) {
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
                    let receiver = console_receiver
                        .as_mut()
                        .ok_or_else(|| {
                            TransferFailure::Infrastructure(
                                "worker emitted unrequested console evidence".into(),
                            )
                        })?;
                    receiver.push(record)
                        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                    completion.record_console(receiver.records().last().expect("retained console record"))
                        .map_err(TransferFailure::Infrastructure)?;
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
                    if let Some(receipt) = &receipt {
                        completion.finish_console(receipt).map_err(TransferFailure::Infrastructure)?;
                    }
                    console_receipt = receipt;
                    break response;
                }
                WorkerProcessMessage::InfrastructureFailure { message } => {
                    // Do not reap the owned Graph leader before the outer
                    // error path retires its group. A completion wait here
                    // would release the PID while descendants remain.
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
        match tokio::time::timeout(Duration::from_secs(5), child.wait_and_retire_tree()).await {
            Ok(Ok(exit)) => {
                child_retired = true;
                *retirement = LocalRetirementEvidence::FullTreeRetired;
                *termination = AttemptExecutionTermination::ProcessExited {
                    code: exit.code,
                    success: exit.success,
                };
            }
            Ok(Err(error)) => {
                let message = format!(
                    "Graph worker replied, but its process tree could not be retired: {error}"
                );
                // RetirementBoundChild already persisted its exact lease and
                // forbids a second numeric group signal after this error.
                child_preserved_reason = Some(message.clone());
                return Err(TransferFailure::RetirementUnverified(message));
            }
            Err(_) => {
                return Err(if completion.is_cancelled() {
                    TransferFailure::Cancelled
                } else {
                    TransferFailure::WorkerLost(
                        "Graph worker did not retire after its response".into(),
                    )
                });
            }
        }
    } else {
        // The leader must stay pinned until its entire owned tree retires.
        drop(writer);
        let exit = match tokio::time::timeout(Duration::from_secs(5), child.wait_and_retire_tree()).await {
            Ok(Ok(exit)) => exit,
            Ok(Err(error)) => return Err(TransferFailure::RetirementUnverified(format!(
                "worker replied, but its process tree could not be retired: {error}"
            ))),
            Err(_) => return Err(TransferFailure::WorkerLost(
                "worker did not retire after its response".into()
            )),
        };
        child_retired = true;
        *retirement = LocalRetirementEvidence::FullTreeRetired;
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
            .graph_authority()
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
    // Console validation must not short-circuit the physical-owner cleanup.
    // Keep the first operation failure; a later cleanup failure takes safety
    // priority while the scheduler separately retains original cancellation.
    let console_result = console_receiver
        .map(|receiver| receiver.finish(console_receipt.as_ref()))
        .transpose()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()));
    let outcome = match console_result {
        Ok(transcript) => {
            *console = transcript;
            outcome
        }
        Err(error) => match outcome {
            Err(original) => Err(original),
            Ok(_) => Err(error),
        },
    };
    if !child_retired {
        if let Some(reason) = child_preserved_reason {
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
                    reason,
                    retired: false,
                    retryable: false,
                });
            return outcome;
        }
        let mut retirement_error = None;
        for _ in 0..2 {
            match child.terminate_and_observe().await {
                Ok(exit) => {
                    *retirement = LocalRetirementEvidence::FullTreeRetired;
                    *termination = AttemptExecutionTermination::ProcessExited {
                        code: exit.code,
                        success: exit.success,
                    };
                    retirement_error = None;
                    break;
                }
                Err(error) => retirement_error = Some(error),
            }
        }
        if let Some(error) = retirement_error {
            let original = outcome.as_ref().err().map_or_else(
                || "worker returned a terminal response".to_owned(),
                ToString::to_string,
            );
            let message = format!(
                "worker process tree could not be retired after bounded retries: {error}; original outcome: {original}"
            );
            if !matches!(&request, WorkerLaunchRequest::StagedGraph(_)) {
                return Err(TransferFailure::RetirementUnverified(message));
            }
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
                    retryable: true,
                });
            return Err(TransferFailure::RetirementUnverified(message));
        }
    }
    outcome
}
