//! The existing owned process body, shared without duplicating Graph retirement.
use super::super::GraphAttemptAuthority;
use super::*;

pub(crate) struct ProcessOwner<'a> {
    pub(crate) attempt: &'a AttemptRequest,
    pub(crate) config: &'a crate::NativeExecutionConfig,
    pub(crate) artifacts: &'a crate::local_store::ArtifactStore,
    pub(crate) objects: &'a crate::FilesystemObjectStore,
    pub(crate) graph_quarantine: &'a std::sync::Mutex<Vec<GraphProcessQuarantine>>,
}
pub(crate) trait OwnedProcessControl:
    crate::process_output::OutputDrainControl + Send + Sync
{
    /// Each caller must provide its original Runtime-managed owner or state
    /// explicitly that its control storage has no such owner.
    fn control_allocation_owner(
        &self,
    ) -> Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>;
    fn fea_provider(&self) -> crate::fea_proxy::Provider {
        crate::fea_proxy::capture()
    }
    fn fea_parent(&self) -> Result<Option<crate::fea_delegation::Source>, String> {
        crate::fea_delegation::capture()
    }
    fn begin_raw_output(
        &self,
        attempt: &AttemptRequest,
        materialized: u64,
    ) -> Result<
        std::sync::Mutex<crate::process_output::RawOutputJournal>,
        std::sync::Arc<crate::RawOutputError>,
    >;
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
impl crate::process_output::OutputDrainControl for LocalControl<'_> {
    fn raw_output_changed(&self) {
        self.completion.raw_output_changed();
    }

    fn stop_for_raw_output(&self, error: std::sync::Arc<crate::RawOutputError>) {
        self.completion.stop_for_raw_output(error);
    }

    fn raw_output_failure(&self) -> Option<std::sync::Arc<crate::RawOutputError>> {
        self.completion.raw_output_failure()
    }
}
impl OwnedProcessControl for LocalControl<'_> {
    fn control_allocation_owner(
        &self,
    ) -> Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner> {
        self.completion.raw_control_owner()
    }
    fn fea_provider(&self) -> crate::fea_proxy::Provider {
        self.completion.fea_provider.clone()
    }
    fn fea_parent(&self) -> Result<Option<crate::fea_delegation::Source>, String> {
        self.completion.fea_parent.clone()
    }
    fn begin_raw_output(
        &self,
        attempt: &AttemptRequest,
        materialized: u64,
    ) -> Result<
        std::sync::Mutex<crate::process_output::RawOutputJournal>,
        std::sync::Arc<crate::RawOutputError>,
    > {
        let root = self
            .driver
            .config
            .store_root
            .join("raw-output")
            .join(attempt.id.to_string());
        let journal =
            crate::process_output::RawOutputJournal::for_attempt(root, attempt, materialized)
                .map_err(std::sync::Arc::new)
                .inspect_err(|error| {
                    self.completion
                        .stop_for_raw_output(std::sync::Arc::clone(error))
                })?;
        self.completion
            .register_raw_output(journal.reader_ref())
            .map_err(std::sync::Arc::new)
            .inspect_err(|error| {
                self.completion
                    .stop_for_raw_output(std::sync::Arc::clone(error))
            })?;
        Ok(std::sync::Mutex::new(journal))
    }
    fn is_cancelled(&self) -> bool {
        self.completion.should_stop()
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

struct RetainedLaunchStorage<T = WorkerLaunchRequest> {
    // Drop the real request backing before refunding its fixed growth claim.
    request: T,
    _layout_claim: Option<runmat_value::ExactHostAllocationClaim>,
}

impl<T> RetainedLaunchStorage<T> {
    fn unclaimed(request: T) -> Self {
        Self {
            request,
            _layout_claim: None,
        }
    }

    fn with_envelope_growth(
        owner: Option<&runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
        make_request: impl FnOnce() -> T,
    ) -> Result<Self, runmat_runtime::context::host_allocation::HostAllocationError> {
        let claim = owner
            .map(|owner| owner.claim(crate::protocol::FEA_LAUNCH_LAYOUT_GROWTH))
            .transpose()?;
        let request = make_request();
        Ok(Self {
            request,
            _layout_claim: claim,
        })
    }
}

impl RetainedLaunchStorage {
    fn delegated(
        request: Box<crate::protocol::ReservedProgramExecutionRequest>,
        console_budget: Option<runmat_execution::console::ConsoleBudget>,
        grant: crate::fea_delegation::WorkerFeaGrant,
        owner: Option<&runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
    ) -> Result<Self, runmat_runtime::context::host_allocation::HostAllocationError> {
        Self::with_envelope_growth(owner, || {
            WorkerLaunchRequest::ReservedFea(Box::new(
                crate::protocol::ReservedFeaLaunch::from_reserved(request, console_budget, grant),
            ))
        })
    }
}

pub(super) async fn run_process(
    driver: &ProcessOwner<'_>,
    mut request: WorkerLaunchRequest,
    completion: &dyn OwnedProcessControl,
    evidence: &mut ProcessAttemptEvidence,
) -> Result<WorkerResponse, TransferFailure> {
    let meters = &evidence.meters;
    let termination = &mut evidence.termination;
    let console = &mut evidence.console;
    let console_scopes = &mut evidence.console_scopes;
    let retirement = &mut evidence.retirement;
    let fea_parent = completion
        .fea_parent()
        .map_err(TransferFailure::Infrastructure)?;
    let invocation = runmat_execution::console::invocation_memory_bytes(
        &driver.attempt.task.environment,
        driver.attempt.task.console,
    )
    .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let mut fea_proxy = if fea_parent.is_none() {
        let minimum =
            crate::fea_proxy::storage_bytes(invocation, crate::fea_proxy::StorageRole::Proxy)
                .map_err(TransferFailure::Infrastructure)?;
        if request
            .resources()
            .is_some_and(|resources| minimum > resources.memory_bytes)
        {
            return Err(TransferFailure::Infrastructure(
                "FEA proxy and invocation controls exceed original worker memory".into(),
            ));
        }
        match (request.assignment(), request.resources()) {
            (Some(assignment), Some(resources)) => Some(
                crate::fea_proxy::Parent::new(
                    completion.fea_provider(),
                    assignment.clone(),
                    resources.clone(),
                    driver.attempt.task.deadline_unix_millis,
                    invocation,
                )
                .map_err(TransferFailure::Infrastructure)?,
            ),
            _ => None,
        }
    } else {
        None
    };
    if fea_proxy.is_some() {
        request = match request {
            WorkerLaunchRequest::Reserved(request) => WorkerLaunchRequest::ReservedFeaProxy {
                request,
                console_budget: None,
                fea_parent_admission: true,
            },
            WorkerLaunchRequest::ReservedConsole {
                request,
                console_budget,
            } => WorkerLaunchRequest::ReservedFeaProxy {
                request,
                console_budget: Some(console_budget),
                fea_parent_admission: true,
            },
            WorkerLaunchRequest::StagedGraph(mut graph) => {
                graph.staged_graph_v10.fea_parent_admission = true;
                WorkerLaunchRequest::StagedGraph(graph)
            }
            other => other,
        };
    }
    let retained_request = if let Some(source) = &fea_parent {
        let assignment = request.assignment().ok_or_else(|| {
            TransferFailure::Infrastructure("FEA child lacks an original assignment".into())
        })?;
        let resources = request.resources().ok_or_else(|| {
            TransferFailure::Infrastructure("FEA child lacks reserved resources".into())
        })?;
        let minimum =
            crate::fea_proxy::storage_bytes(invocation, crate::fea_proxy::StorageRole::Delegated)
                .map_err(TransferFailure::Infrastructure)?;
        if resources.memory_bytes < minimum {
            return Err(TransferFailure::Infrastructure(
                "inherited FEA controls exceed original worker memory".into(),
            ));
        }
        let (mut grant, grant_owner) =
            crate::fea_delegation::ChildGrant::new(source, assignment, resources)
                .map_err(TransferFailure::Infrastructure)?;
        fea_proxy = Some(
            crate::fea_proxy::Parent::from_delegation(
                &mut grant,
                grant_owner,
                driver.attempt.task.deadline_unix_millis,
                invocation,
            )
            .map_err(TransferFailure::Infrastructure)?,
        );
        let owner = completion.control_allocation_owner();
        let retained = match request {
            WorkerLaunchRequest::Reserved(request) => {
                RetainedLaunchStorage::delegated(request, None, grant, owner.as_ref())
            }
            WorkerLaunchRequest::ReservedConsole {
                request,
                console_budget,
            } => RetainedLaunchStorage::delegated(
                request,
                Some(console_budget),
                grant,
                owner.as_ref(),
            ),
            WorkerLaunchRequest::StagedGraph(mut graph) => {
                graph.staged_graph_v10.fea_grant = Some(grant);
                Ok(RetainedLaunchStorage::unclaimed(
                    WorkerLaunchRequest::StagedGraph(graph),
                ))
            }
            _ => {
                return Err(TransferFailure::Infrastructure(
                    "FEA delegation requires the private reserved launch".into(),
                ))
            }
        };
        retained.map_err(super::control_allocation_failure)?
    } else {
        RetainedLaunchStorage::unclaimed(request)
    };
    let request = &retained_request.request;
    let _fea_proxy_close = fea_proxy
        .as_ref()
        .map(|proxy| crate::fea_proxy::Close(std::sync::Arc::clone(proxy)));
    #[cfg(windows)]
    if matches!(request, WorkerLaunchRequest::StagedGraph(_)) {
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
    match request {
        WorkerLaunchRequest::Direct(request) => request
            .validate()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
        WorkerLaunchRequest::Reserved(request)
        | WorkerLaunchRequest::ReservedFeaProxy { request, .. }
        | WorkerLaunchRequest::ReservedConsole { request, .. } => request
            .validate()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
        WorkerLaunchRequest::ReservedFea(launch) => launch
            .request
            .validate()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
        WorkerLaunchRequest::StagedGraph(_) => {}
    }
    let console_controls = match request {
        WorkerLaunchRequest::Reserved(request)
        | WorkerLaunchRequest::ReservedFeaProxy { request, .. }
        | WorkerLaunchRequest::ReservedConsole { request, .. } => request.request().console,
        WorkerLaunchRequest::ReservedFea(launch) => launch.request.request().console,
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
    let console_budget = request.console_budget();
    if console_controls.is_some() != console_budget.is_some() {
        return Err(TransferFailure::Infrastructure(
            "worker capture omits its original delegated subtree grant".into(),
        ));
    }
    if let Some(budget) = console_budget {
        budget
            .validate()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    }
    let mut console_receiver = console_controls
        .map(runmat_execution::console::ConsoleReceiver::new)
        .transpose()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let mut console_receipt = None;
    let mut received_scopes = Vec::new();
    let payload = serde_json::to_vec(request)
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    if payload.len() > driver.config.max_message_bytes as usize {
        return Err(TransferFailure::Infrastructure(
            "worker request exceeds the native message bound".into(),
        ));
    }
    if completion.is_cancelled() {
        return Err(TransferFailure::Cancelled);
    }
    let graph_scratch = if matches!(request, WorkerLaunchRequest::StagedGraph(_)) {
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
    let listener = runmat_process_host::ipc::private_control::PrivateControlListener::bind()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    command.stdio = runmat_process_host::StdioPolicy::PrivateControl(listener.endpoint().clone());
    command.arguments = driver.config.worker_arguments.clone();
    command.environment_policy = EnvironmentPolicy::Inherit;
    if let Some(org) =
        crate::config::local_organization_environment(driver.config.local_fea_organization)
    {
        // This owned local worker resolves the same explicit CLI account selector.
        // The signed-in person's membership remains Server-authorized.
        command.environment.insert("RUNMAT_ORG_ID".into(), org);
    }
    if matches!(request, WorkerLaunchRequest::StagedGraph(_)) {
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
    if matches!(request, WorkerLaunchRequest::StagedGraph(_)) {
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
    if let Some(proxy) = &fea_proxy {
        proxy
            .may_execute()
            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    }
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
    let materialized = match request {
        WorkerLaunchRequest::StagedGraph(_) => completion
            .graph_authority()
            .map(|authority| authority.materialized_scratch_bytes),
        _ => Some(0),
    };
    let mut raw_journal = None;
    let mut output_pipes = None;
    let setup_error = match materialized {
        None => Some(TransferFailure::Infrastructure(
            "Graph raw output lacks its admitted materialization authority".into(),
        )),
        Some(materialized) => match completion.begin_raw_output(driver.attempt, materialized) {
            Err(error) => {
                completion.stop_for_raw_output(std::sync::Arc::clone(&error));
                // This attempt's output owner has stopped. Retrying the same
                // task cannot clear that owner or enlarge its original grant.
                Some(TransferFailure::Execution(error.to_string()))
            }
            Ok(journal) => {
                raw_journal = Some(journal);
                match child.take_application_output() {
                    Ok(pipes) => {
                        output_pipes = Some(pipes);
                        None
                    }
                    Err(error) => {
                        let mut failure = std::sync::Arc::new(crate::RawOutputError::Io(
                            std::io::Error::other(error),
                        ));
                        if let Some(journal) = raw_journal.as_ref() {
                            let mut journal = journal.lock().expect("raw output journal poisoned");
                            if let Err(cleanup) = journal.fail_open_streams() {
                                failure = std::sync::Arc::new(crate::RawOutputError::Cleanup {
                                    primary: failure,
                                    cleanup: Box::new(cleanup),
                                });
                            }
                        }
                        completion.stop_for_raw_output(std::sync::Arc::clone(&failure));
                        completion.raw_output_changed();
                        Some(TransferFailure::Execution(failure.to_string()))
                    }
                }
            }
        },
    };
    let output_lifetime = crate::process_output::OutputDrainLifetime::default();
    let stderr = child.captured_stderr();
    let mut deferred_exchange = DeferredWorkerExchangeFailure::default();
    let owner = async {
        let result = async {
    let mut child_retired = false;
    let mut child_preserved_reason = None;
    let outcome = async {
    if let Some(error) = setup_error {
        return Err(error);
    }
    if completion.is_cancelled() {
        return Err(TransferFailure::Cancelled);
    }
    // Application stdout/stderr require a caller-owned drain before this
    // bootstrap is qualified; neither stream is a control transport.
    let accept = tokio::time::timeout(Duration::from_secs(30), listener.accept());
    tokio::pin!(accept);
    let control = loop {
        tokio::select! {
            connected = &mut accept => break connected
                .map_err(|_| TransferFailure::Infrastructure(
                    "native worker private control bootstrap exceeded its deadline".into(),
                ))?
                .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if completion.is_cancelled() {
                    return Err(TransferFailure::Cancelled);
                }
            }
        }
    };
    if completion.is_cancelled() {
        return Err(TransferFailure::Cancelled);
    }
    let mut reader = BufReader::new(MeteredIo::new(control.reader, meters.received.clone()));
    let mut writer = MeteredIo::new(control.writer, meters.sent.clone());
    let limits = FrameLimits {
        max_message_bytes: driver.config.max_message_bytes,
    };
    {
        let write = write_payload(&mut writer, &payload, limits);
        tokio::pin!(write);
        loop {
            tokio::select! {
                result = &mut write => {
                    result.map_err(|error| deferred_exchange.record(error))?;
                    break;
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    if completion.is_cancelled() {
                        return Err(TransferFailure::Cancelled);
                    }
                }
            }
        }
    }
    if completion.is_cancelled() {
        return Err(TransferFailure::Cancelled);
    }
    let mut last_progress_sequence = 0;
    let mut last_fea_budget = request.fea_grant().map(|grant| grant.parent.budget);
    let mut graph_cancel_sent_at = None;
    let response = loop {
        let payload = {
        // Keep this same read alive across cancellation ticks: read_exact can
        // already have consumed part of the header or body before Pending.
        let read = read_payload(&mut reader, limits);
        tokio::pin!(read);
        loop { tokio::select! {
            response = &mut read => {
                break response.map_err(|error| deferred_exchange.record(error))?;
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if let Some(proxy) = &fea_proxy { proxy.check().map_err(|denied| TransferFailure::Infrastructure(denied.to_string()))?; }
                if let Some(source) = &fea_parent {
                    let budget = match fea_proxy.as_ref().ok_or_else(|| TransferFailure::Infrastructure("FEA child lost its retained grant owner".into()))?.delegated_budget() {
                        Ok(budget) => budget,
                        Err(denied) => {
                            if let Some(grant) = request.fea_grant() {
                                let update = WorkerDriverMessage::FeaLeaseUpdate { run_id: grant.parent.request.run_id.clone(), attempt_id: grant.assignment.attempt_id,
                                    update: crate::fea_delegation::FeaLeaseUpdate::Denied { denied: denied.clone() } };
                                if let Ok(payload) = serde_json::to_vec(&update) { let _ = tokio::time::timeout(Duration::from_secs(1), write_payload(&mut writer, &payload, limits)).await; }
                            }
                            return Err(TransferFailure::Infrastructure(denied.to_string()));
                        }
                    };
                    if source.cancellation_token().is_cancelled() { return Err(TransferFailure::Cancelled); }
                    if last_fea_budget != Some(budget) {
                        let grant = request.fea_grant().ok_or_else(|| TransferFailure::Infrastructure("FEA renewal lacks its original child grant".into()))?;
                        let update = WorkerDriverMessage::FeaLeaseUpdate { run_id: grant.parent.request.run_id.clone(), attempt_id: grant.assignment.attempt_id,
                            update: crate::fea_delegation::FeaLeaseUpdate::Renewed { budget } };
                        let payload = serde_json::to_vec(&update).map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                        tokio::time::timeout(Duration::from_secs(1), write_payload(&mut writer, &payload, limits)).await
                            .map_err(|_| TransferFailure::Infrastructure("FEA renewal delivery exceeded its bound".into()))?
                            .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                        last_fea_budget = Some(budget);
                    }
                }
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
        if payload.len() as u64 <= crate::fea_proxy::MAX_FRAME_BYTES && crate::fea_proxy::is_call(&payload) && fea_proxy.is_none() {
            return Err(TransferFailure::Infrastructure("worker requested ungranted FEA admission proxy".into()));
        }
        if let Ok(message) = serde_json::from_slice::<WorkerProcessMessage>(&payload) {
            match message {
                WorkerProcessMessage::FeaAdmissionCall { call } => {
                    if payload.len() as u64 > crate::fea_proxy::MAX_FRAME_BYTES { return Err(TransferFailure::Infrastructure("FEA proxy request exceeds its control bound".into())); }
                    let proxy = fea_proxy.as_ref().ok_or_else(|| TransferFailure::Infrastructure("worker requested ungranted FEA admission proxy".into()))?.clone();
                    let sequence = call.sequence;
                    let worker = proxy.clone();
                    let operation = tokio::task::spawn_blocking(move || worker.call(call));
                    tokio::pin!(operation);
                    let reply = loop { tokio::select! {
                        reply = &mut operation => break reply.map_err(|error| TransferFailure::Infrastructure(error.to_string()))?,
                        _ = tokio::time::sleep(Duration::from_millis(10)) => {
                            if completion.is_cancelled() { return Err(TransferFailure::Cancelled); }
                            proxy.check().map_err(|denied| TransferFailure::Infrastructure(denied.to_string()))?;
                        }
                    }};
                    let encoded = runmat_canonical_codec::encode_json_bounded(&WorkerDriverMessage::FeaAdmissionReply { sequence, reply }, crate::fea_proxy::MAX_FRAME_BYTES)
                        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                    tokio::time::timeout(Duration::from_secs(1), write_payload(&mut writer, &encoded, limits)).await
                        .map_err(|_| TransferFailure::Infrastructure("FEA proxy reply delivery exceeded its bound".into()))?
                        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                }
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
                    {
                        let write = write_payload(&mut writer, &payload, limits);
                        tokio::pin!(write);
                        loop {
                            tokio::select! {
                                result = &mut write => {
                                    result.map_err(|error| deferred_exchange.record(error))?;
                                    break;
                                }
                                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                                    if completion.is_cancelled() {
                                        return Err(TransferFailure::Cancelled);
                                    }
                                }
                            }
                        }
                    }
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
                    console_scopes: scopes,
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
                    received_scopes = scopes;
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
    if matches!(request, WorkerLaunchRequest::StagedGraph(_)) {
        // This is a one-shot Graph child. Close our private request writer
        // after the terminal frame so the worker observes EOF before wait.
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
    if let Some(source) = &fea_parent {
        source.current_budget().map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        if source.cancellation_token().is_cancelled() { return Err(TransferFailure::Cancelled); }
    }
    if let Some(proxy) = &fea_proxy { proxy.terminal_check().map_err(|denied| TransferFailure::Infrastructure(denied.to_string()))?; }
    match request {
        WorkerLaunchRequest::Direct(request) => match request.as_ref() {
            WorkerRequest::Direct(request) => response.validate_against(request),
            WorkerRequest::Graph(_) => {
            return Err(TransferFailure::Infrastructure(
                "Graph worker response has no staged authority".into(),
            ));
            }
        },
        WorkerLaunchRequest::Reserved(request)
        | WorkerLaunchRequest::ReservedFeaProxy { request, .. }
        | WorkerLaunchRequest::ReservedConsole { request, .. } => response.validate_against(request.request()),
        WorkerLaunchRequest::ReservedFea(launch) => response.validate_against(launch.request.request()),
        WorkerLaunchRequest::StagedGraph(_) => completion
            .graph_authority()
            .ok_or_else(|| {
                TransferFailure::Infrastructure("Graph response lacks its typed authority".into())
            })?
            .response
            .validate_response(&response),
    }
    .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    if graph_cancel_sent_at.is_some() || completion.cancellation_reason().is_some() {
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
            match crate::console_bridge::validate_imported_subtree(console.as_ref(), &received_scopes, console_budget) {
                Ok(()) => {
                    *console_scopes = received_scopes;
                    outcome
                }
                Err(message) => {
                    deferred_exchange.clear();
                    Err(TransferFailure::Infrastructure(message))
                },
            }
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
            if !matches!(request, WorkerLaunchRequest::StagedGraph(_)) {
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
        .await;
        output_lifetime.finish();
        result
    };
    let drain = async {
        match raw_journal.as_ref() {
            Some(journal) if output_pipes.is_some() => {
                crate::process_output::drain_pipes(
                    output_pipes,
                    journal,
                    completion,
                    &output_lifetime,
                )
                .await
            }
            _ => Ok(()),
        }
    };
    let (outcome, drain_result) = tokio::join!(owner, drain);
    let outcome = if *retirement == LocalRetirementEvidence::FullTreeRetired {
        match fea_proxy
            .as_ref()
            .map(|proxy| proxy.retire_delegated(outcome.is_ok()))
            .transpose()
        {
            Ok(_) => outcome,
            Err(error) => {
                if matches!(
                    &outcome,
                    Err(_)
                        | Ok(WorkerResponse::Failure { .. })
                        | Ok(WorkerResponse::RuntimeFailure { .. })
                ) {
                    // Retain the original failure's type, identifier and source
                    // frames. The delegated hold remains owned by its ledger;
                    // this diagnostic cannot certify incomplete accounting.
                    tracing::error!(
                        admission_code = error.error_code(),
                        error = %error,
                        "Delegated simulation accounting cleanup failed after worker failure"
                    );
                    outcome
                } else {
                    Err(TransferFailure::Infrastructure(error.to_string()))
                }
            }
        }
    } else {
        outcome
    };
    // Finish existing output/retirement precedence before attaching the final
    // original transport diagnostic. This closure has no producer or await.
    let outcome = (|| {
        let retirement_result = if *retirement == LocalRetirementEvidence::FullTreeRetired {
            raw_journal.as_ref().map(|journal| {
                journal
                    .lock()
                    .expect("raw output journal poisoned")
                    .mark_retired()
            })
        } else {
            None
        };
        let retirement_error = retirement_result.and_then(Result::err);
        let has_retirement_cleanup_error = retirement_error.is_some();
        let raw_error = match (drain_result.err(), retirement_error) {
            (Some(primary), Some(cleanup)) => {
                Some(std::sync::Arc::new(crate::RawOutputError::Cleanup {
                    primary,
                    cleanup: Box::new(cleanup),
                }))
            }
            (None, Some(cleanup)) => Some(match completion.raw_output_failure() {
                Some(primary) => std::sync::Arc::new(crate::RawOutputError::Cleanup {
                    primary,
                    cleanup: Box::new(cleanup),
                }),
                None => std::sync::Arc::new(cleanup),
            }),
            (Some(primary), None) => Some(primary),
            (None, None) => None,
        };
        if let Some(error) = raw_error {
            completion.stop_for_raw_output(std::sync::Arc::clone(&error));
            if matches!(
                &outcome,
                Ok(WorkerResponse::Success { .. })
                    | Ok(WorkerResponse::ExternalizedSuccess { .. })
                    | Ok(WorkerResponse::GraphSuccess { .. })
                    | Ok(WorkerResponse::SpmdSuccess { .. })
                    | Err(TransferFailure::Cancelled)
            ) {
                return Err(TransferFailure::Execution(error.to_string()));
            }
            if has_retirement_cleanup_error {
                return match outcome {
                    Err(TransferFailure::Infrastructure(message)) => {
                        Err(TransferFailure::Infrastructure(format!(
                            "{message}; output cleanup also failed: {error}"
                        )))
                    }
                    Err(TransferFailure::RetirementUnverified(message)) => {
                        Err(TransferFailure::RetirementUnverified(format!(
                            "{message}; output cleanup also failed: {error}"
                        )))
                    }
                    other => other,
                };
            }
        }
        outcome
    })();
    deferred_exchange.finish(outcome, &stderr)
}

#[cfg(test)]
mod layout_claim_tests {
    use super::RetainedLaunchStorage;
    use runmat_runtime::context::host_allocation::HostControlAllocationOwner;
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
    use runmat_runtime::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };
    use std::{cell::Cell, rc::Rc};

    struct Parallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for Parallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn launch_backing_retires_before_original_growth_claim_is_refunded() {
        let ledger = ManagedHostEffectLedger::new_with_epoch(4096, 7);
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        let owner = HostControlAllocationOwner::capture(&runtime)
            .unwrap()
            .unwrap();
        struct Backing {
            ledger: ManagedHostEffectLedger,
            dropped: Rc<Cell<bool>>,
        }
        impl Drop for Backing {
            fn drop(&mut self) {
                assert!(self.ledger.snapshot().live_claimed_bytes > 0);
                self.dropped.set(true);
            }
        }
        let dropped = Rc::new(Cell::new(false));
        let retained = RetainedLaunchStorage::with_envelope_growth(Some(&owner), || {
            assert!(
                ledger.snapshot().live_claimed_bytes
                    >= crate::protocol::FEA_LAUNCH_LAYOUT_GROWTH as u64
            );
            Box::new(Backing {
                ledger: ledger.clone(),
                dropped: dropped.clone(),
            })
        })
        .unwrap();
        assert!(!dropped.get());
        assert!(
            ledger.snapshot().live_claimed_bytes
                >= crate::protocol::FEA_LAUNCH_LAYOUT_GROWTH as u64
        );
        drop(retained);
        assert!(dropped.get());
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let factory_called = Cell::new(false);
        assert!(
            RetainedLaunchStorage::with_envelope_growth(Some(&owner), || {
                factory_called.set(true);
            })
            .is_err()
        );
        assert!(
            !factory_called.get(),
            "revoked original owner must refuse before constructing backing"
        );
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
