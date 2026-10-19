use std::collections::{BTreeSet, HashMap};
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::state::TaskState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::{GraphRequiredSchema, ValuePayload};
use runmat_execution::{
    CancellationReason, Digest, ExecutionScopeId, OutputContract, PoolId, ProgramCallable,
    ProgramInvocationContext, TaskId,
};
use runmat_execution_artifact::{
    GraphExecutionRequest, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionDescriptor,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, TaskSubmission,
};

pub(crate) mod collective;
mod process;

mod clock;
mod completion;
mod persistence;
pub(crate) use completion::{next_task_completion_order, TaskCompletion};

use crate::local_store::{prepare_session_root, ArtifactStore, CheckpointStore};
use crate::protocol::{StoredDirectProgram, StoredProgram};
use crate::{
    FilesystemObjectStore, NativeExecutionConfig, NativeExecutionError, NativeExecutionResult,
};

pub const NATIVE_ARTIFACT_STORE_ROOT_ENV: &str = "RUNMAT_EXECUTION_ARTIFACT_STORE_ROOT";
pub const NATIVE_GRAPH_ATTEMPT_SCRATCH_ENV: &str = "RUNMAT_GRAPH_ATTEMPT_SCRATCH_ROOT";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TransferFailure {
    Execution(String),
    Infrastructure(String),
    WorkerLost(String),
    RetirementUnverified(String),
    Cancelled,
    Runtime(Box<runmat_execution::ProgramRuntimeFailure>),
}

impl std::fmt::Display for TransferFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution(message)
            | Self::Infrastructure(message)
            | Self::WorkerLost(message)
            | Self::RetirementUnverified(message) => formatter.write_str(message),
            Self::Cancelled => formatter.write_str("execution was cancelled"),
            Self::Runtime(failure) => formatter.write_str(&failure.message),
        }
    }
}

pub(crate) type TransferResult = Result<AttemptSuccess, TransferFailure>;

pub(crate) struct LocalProgramSubmission {
    pub(crate) task_id: TaskId,
    pub(crate) callable: ProgramCallable,
    pub(crate) invocation_context: runmat_execution::ProgramInvocationContext,
    pub(crate) recipe: ProgramBuildRecipe,
    pub(crate) artifact: ProgramArtifact,
    pub(crate) inputs: Vec<ValuePayload>,
    pub(crate) outputs: OutputContract,
    pub(crate) retry: RetryPolicy,
    pub(crate) environment: runmat_execution::EnvironmentSnapshot,
    pub(crate) console: Option<runmat_execution::console::ConsoleLimits>,
}

pub(crate) struct LocalGraphSubmission {
    pub(crate) task_id: TaskId,
    pub(crate) request: GraphExecutionRequest,
    pub(crate) retry: RetryPolicy,
}

pub(crate) struct LocalDriver {
    config: NativeExecutionConfig,
    clock: clock::DriverClock,
    store_lifetime: Arc<crate::program_object_source::SessionStoreLifetime>,
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    driver: Mutex<Driver>,
    artifacts: ArtifactStore,
    objects: FilesystemObjectStore,
    checkpoints: CheckpointStore,
    checkpoint_evidence: Mutex<crate::NativeCheckpointEvidence>,
    /// An unverified Graph process tree keeps its exact parent-owned scratch
    /// and child handle until shutdown; new work is refused while unhealthy.
    graph_quarantine: Mutex<Vec<process::GraphProcessQuarantine>>,
    /// A signed one-worker remote allocation may differ from the ordinary
    /// local-service inventory. Reject an unschedulable Graph at admission.
    remote_graph_inventory: Option<ResourceInventory>,
    remote_graph_max_wall_millis: Option<u64>,
    completions: Mutex<HashMap<TaskId, Arc<TaskCompletion>>>,
    collectives: collective::ProcessCollectiveBroker,
    #[cfg(test)]
    fail_next_checkpoint: AtomicBool,
}

impl LocalDriver {
    pub(crate) fn new(
        config: NativeExecutionConfig,
        scope_id: ExecutionScopeId,
    ) -> NativeExecutionResult<Arc<Self>> {
        Self::new_with_remote_graph_inventory(config, scope_id, None, None)
    }

    pub(crate) fn new_with_remote_graph_inventory(
        config: NativeExecutionConfig,
        scope_id: ExecutionScopeId,
        remote_graph_inventory: Option<ResourceInventory>,
        remote_graph_max_wall_millis: Option<u64>,
    ) -> NativeExecutionResult<Arc<Self>> {
        config
            .validate()
            .map_err(NativeExecutionError::Configuration)?;
        if let Some(inventory) = &remote_graph_inventory {
            inventory
                .validate()
                .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?;
            if config.max_workers != 1 {
                return Err(NativeExecutionError::Configuration(
                    "remote Graph inventory requires one owned attempt worker".into(),
                ));
            }
        }
        if remote_graph_max_wall_millis == Some(0) {
            return Err(NativeExecutionError::Configuration(
                "remote Graph wall-clock allocation must be nonzero".into(),
            ));
        }
        let mut driver = Driver::new(DriverConfig::default(), 1)?;
        let pool_id = crate::native_pool::register_with_inventory(
            &mut driver,
            &config,
            scope_id,
            remote_graph_inventory.as_ref(),
        )?;
        prepare_session_root(&config.store_root)?;
        let store_lifetime =
            crate::program_object_source::SessionStoreLifetime::new(config.store_root.clone());
        let artifacts = ArtifactStore::new(config.store_root.join("artifacts"))?;
        let objects =
            FilesystemObjectStore::open(config.store_root.join("objects"), config.max_object_bytes)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let checkpoints = CheckpointStore::new(config.store_root.join("checkpoints"))?;
        let clock = clock::DriverClock::new()?;
        driver.handle(DriverCommand::Tick {
            now_millis: clock.now_millis(),
        })?;
        let local = Arc::new(Self {
            store_lifetime,
            clock,
            config,
            scope_id,
            pool_id,
            driver: Mutex::new(driver),
            artifacts,
            objects,
            checkpoints,
            checkpoint_evidence: Mutex::new(crate::NativeCheckpointEvidence::NotPublished {
                diagnostic: "initial snapshot has not been published".into(),
            }),
            graph_quarantine: Mutex::new(Vec::new()),
            remote_graph_inventory,
            remote_graph_max_wall_millis,
            completions: Mutex::new(HashMap::new()),
            collectives: collective::ProcessCollectiveBroker::default(),
            #[cfg(test)]
            fail_next_checkpoint: AtomicBool::new(false),
        });
        local.checkpoint()?;
        clock::start(&local)?;
        Ok(local)
    }

    pub(crate) fn object_store(&self) -> FilesystemObjectStore {
        self.objects.clone()
    }

    pub(crate) fn object_source(&self) -> crate::NativeProgramObjectSource {
        crate::NativeProgramObjectSource::new(
            self.objects.clone(),
            Arc::clone(&self.store_lifetime),
        )
    }

    pub(crate) const fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    pub(crate) const fn pool_id(&self) -> PoolId {
        self.pool_id
    }

    pub(crate) const fn max_workers(&self) -> u32 {
        self.config.max_workers
    }

    pub(crate) fn active_workers(&self) -> NativeExecutionResult<u32> {
        let driver = self.driver.lock().expect("local driver poisoned");
        let pool = driver
            .snapshot()
            .pools
            .get(&self.pool_id)
            .cloned()
            .ok_or_else(|| NativeExecutionError::Protocol("local pool disappeared".into()))?;
        u32::try_from(
            pool.workers
                .values()
                .filter(|worker| worker.accepts_work())
                .count(),
        )
        .map_err(|_| NativeExecutionError::Protocol("local worker count exceeds u32".into()))
    }

    pub(crate) fn resize_pool(self: &Arc<Self>, desired_workers: u32) -> NativeExecutionResult<()> {
        let actions = self
            .driver
            .lock()
            .expect("local driver poisoned")
            .resize_registered_pool(self.pool_id, desired_workers)?;
        self.checkpoint()?;
        Self::dispatch(Arc::clone(self), actions);
        Ok(())
    }

    pub(crate) fn submit(
        self: &Arc<Self>,
        submission: LocalProgramSubmission,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        let LocalProgramSubmission {
            task_id,
            callable,
            invocation_context,
            recipe,
            artifact,
            inputs,
            outputs,
            retry,
            environment,
            console,
        } = submission;
        let context_bytes = runmat_execution::console::invocation_memory_bytes(&environment, console)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let memory_bytes = (1024_u64 * 1024)
            .checked_add(context_bytes)
            .ok_or_else(|| {
                NativeExecutionError::Protocol("invocation memory reservation overflow".into())
            })?;
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let host = artifact
            .execution_host_requirement(
                &recipe,
                BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
            )
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let accelerator_requirements = recipe.accelerators.clone();
        let request = TaskRequest {
            id: task_id,
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact_id),
            callable: Callable::for_program("local-session", &callable),
            host,
            invocation_context,
            inputs,
            environment,
            console,
            outputs,
            resources: ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes,
                scratch_bytes: 1024 * 1024,
                max_wall_millis: 24 * 60 * 60 * 1000,
                max_artifact_bytes: u64::from(self.config.max_message_bytes),
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: accelerator_requirements,
                required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            },
            retry,
            deadline_unix_millis: None,
        };
        self.submit_task(
            TaskSubmission {
                request,
                dependencies: BTreeSet::new(),
                priority: 0,
            },
            recipe,
            artifact,
        )
    }

    pub(crate) fn submit_task(
        self: &Arc<Self>,
        submission: TaskSubmission,
        recipe: ProgramBuildRecipe,
        artifact: ProgramArtifact,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        self.submit_tasks(vec![(submission, recipe, artifact)])?
            .pop()
            .ok_or_else(|| {
                NativeExecutionError::Protocol("local task admission returned no completion".into())
            })
    }

    /// Atomically admits a topologically ordered batch before dispatching any task.
    pub(crate) fn submit_tasks(
        self: &Arc<Self>,
        tasks: Vec<(TaskSubmission, ProgramBuildRecipe, ProgramArtifact)>,
    ) -> NativeExecutionResult<Vec<Arc<TaskCompletion>>> {
        self.ensure_no_graph_quarantine()?;
        let mut registrations = Vec::with_capacity(tasks.len());
        for (submission, recipe, artifact) in tasks {
            ProgramExecutionDescriptor {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe: recipe.clone(),
                artifact: artifact.clone(),
                callable: submission.request.callable.program.clone(),
                requested_outputs: submission.request.outputs.requested_outputs,
                environment: submission.request.environment.clone(),
                console: submission.request.console,
            }
            .validate_for_portable_host()
            .map_err(|error| {
                NativeExecutionError::Protocol(format!(
                    "local program descriptor failed validation: {error}"
                ))
            })?;
            let stored_program =
                StoredProgram::Direct(Box::new(StoredDirectProgram { recipe, artifact }));
            let encoded = serde_json::to_vec(&stored_program)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            self.stage_stored_task(&submission, &stored_program, &encoded)?;
            registrations.push((submission, Arc::new(TaskCompletion::new())));
        }
        self.publish_submissions(registrations)
    }

    pub(crate) fn submit_graph(
        self: &Arc<Self>,
        graph: LocalGraphSubmission,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        self.ensure_no_graph_quarantine()?;
        let admitted = graph
            .request
            .into_validated()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let selected = admitted.request();
        if selected.assignment.is_some() || selected.job_id.is_some() {
            return Err(NativeExecutionError::Protocol(
                "worker graph assignment is owned by placement".into(),
            ));
        }
        let callable = selected
            .scheduler_callable()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let host = selected
            .execution_host_requirement(BTreeSet::from([self.config.host_inventory.trust_tier]))
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        // The untagged StoredProgram::Graph wire shape is precisely this one
        // field. Borrowing avoids a second Graph payload clone before writing
        // it once into the parent's private artifact store.
        #[derive(serde::Serialize)]
        struct BorrowedStoredGraph<'a> {
            request: &'a GraphExecutionRequest,
        }
        let stored = self.artifacts.put_streamed_graph(
            &BorrowedStoredGraph { request: selected },
            runmat_execution::resource::MAX_GRAPH_PROGRAM_WIRE_BYTES,
        )?;
        let serialized_length = usize::try_from(stored.length).map_err(|_| {
            NativeExecutionError::Protocol(
                "graph wire length exceeds this host's address space".into(),
            )
        })?;
        let memory_bytes = selected
            .input_allocation_floor_bytes(serialized_length)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let scratch_bytes = stored.length.max(1024 * 1024);
        let recipe = &selected.recipe;
        let artifact_id = stored.id;
        let response_authority = admitted
            .response_authority()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let invocation_context = selected.context.clone();
        let requested_outputs = selected.requested_outputs;
        let mut required_capabilities = BTreeSet::from([Capability::ProcessIsolation]);
        let selected_schema = GraphRequiredSchema::from_version(selected.schema_version)
            .ok_or_else(|| NativeExecutionError::Protocol("unsupported Graph schema".into()))?;
        if let Some(capability) = selected_schema.required_capability() {
            required_capabilities.insert(capability);
        }
        let request = TaskRequest {
            id: graph.task_id,
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact_id),
            callable: Callable::for_program("local-session", &callable),
            host,
            invocation_context: invocation_context.clone(),
            inputs: Vec::new(),
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console: None,
            outputs: recipe.outputs.clone(),
            resources: ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes,
                scratch_bytes,
                max_wall_millis: self
                    .remote_graph_max_wall_millis
                    .unwrap_or(24 * 60 * 60 * 1000),
                max_artifact_bytes: stored.length,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: recipe.accelerators.clone(),
                required_capabilities,
            },
            retry: graph.retry,
            deadline_unix_millis: None,
        };
        if let Some(inventory) = &self.remote_graph_inventory {
            if request.resources.cpu_millicores > inventory.cpu_millicores
                || request.resources.memory_bytes > inventory.memory_bytes
                || request.resources.scratch_bytes > inventory.scratch_bytes
            {
                return Err(NativeExecutionError::Protocol(
                    "graph exceeds the authenticated remote driver allocation".into(),
                ));
            }
        }
        request
            .callable
            .program
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        request
            .invocation_context
            .validate_for(&request.callable.program)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        recipe
            .validate_resource_request(&request.resources)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let completion = Arc::new(TaskCompletion::new_graph(
            stored,
            invocation_context,
            requested_outputs,
            callable,
            response_authority,
        ));
        self.publish_submissions(vec![(
            TaskSubmission {
                request,
                dependencies: BTreeSet::new(),
                priority: 0,
            },
            completion,
        )])?
        .pop()
        .ok_or_else(|| {
            NativeExecutionError::Protocol("local graph admission returned no completion".into())
        })
    }

    /// Validate one stored program against its placement and stage its bytes.
    fn stage_stored_task(
        &self,
        submission: &TaskSubmission,
        stored_program: &StoredProgram,
        stored: &[u8],
    ) -> NativeExecutionResult<()> {
        let (recipe, artifact) = match stored_program {
            StoredProgram::Direct(stored) => (&stored.recipe, &stored.artifact),
            StoredProgram::Graph(stored) => (&stored.request.recipe, &stored.request.artifact),
        };
        recipe
            .validate_resource_request(&submission.request.resources)
            .map_err(|error| {
                NativeExecutionError::Protocol(format!(
                    "local scheduler resources do not satisfy the program: {error}"
                ))
            })?;
        if stored.len() > self.config.max_message_bytes as usize {
            return Err(NativeExecutionError::Protocol(
                "stored worker request exceeds the message bound".into(),
            ));
        }
        let artifact_id = match stored_program {
            StoredProgram::Direct(_) => ArtifactId::derive(&[artifact.id.0.bytes()]),
            StoredProgram::Graph(_) => ArtifactId::derive(&[b"graph-worker-request-v8", stored]),
        };
        if submission.request.scope_id != self.scope_id
            || submission.request.pool_id != self.pool_id
            || submission.request.program_artifact_id() != Some(artifact_id)
            || submission
                .request
                .invocation_context
                .validate_for(&submission.request.callable.program)
                .is_err()
        {
            return Err(NativeExecutionError::Protocol(
                "local task submission differs from its session or program artifact".into(),
            ));
        }
        if let StoredProgram::Graph(stored) = stored_program {
            if submission.request.callable.program
                != stored
                    .request
                    .scheduler_callable()
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?
                || submission.request.invocation_context != stored.request.context
                || submission.request.outputs.requested_outputs != stored.request.requested_outputs
                || !submission.request.inputs.is_empty()
            {
                return Err(NativeExecutionError::Protocol(
                    "graph task placement differs from its selected graph request".into(),
                ));
            }
        }
        submission
            .request
            .callable
            .program
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        self.artifacts.put(artifact_id, stored)
    }

    /// Register and admit a batch atomically: either every task is accepted
    /// and its snapshot published, or the scheduler and registry are restored.
    fn publish_submissions(
        self: &Arc<Self>,
        registrations: Vec<(TaskSubmission, Arc<TaskCompletion>)>,
    ) -> NativeExecutionResult<Vec<Arc<TaskCompletion>>> {
        if registrations.is_empty() {
            return Err(NativeExecutionError::Protocol(
                "local task submission batch must not be empty".into(),
            ));
        }
        // Recheck after staging: a different attempt may have entered
        // quarantine while this submission was being encoded.
        self.ensure_no_graph_quarantine()?;
        let mut completions = self
            .completions
            .lock()
            .expect("completion registry poisoned");
        let mut unique = BTreeSet::new();
        if registrations.iter().any(|(submission, _)| {
            !unique.insert(submission.request.id)
                || completions.contains_key(&submission.request.id)
        }) {
            return Err(NativeExecutionError::Protocol(
                "local task batch contains a duplicate or registered task".into(),
            ));
        }
        for (submission, completion) in &registrations {
            completions.insert(submission.request.id, Arc::clone(completion));
        }
        let (actions, inherited_terminals, checkpoint) = {
            let mut driver = self.driver.lock().expect("local driver poisoned");
            let prior = driver.snapshot();
            let submissions = registrations
                .iter()
                .map(|(submission, _)| submission.clone())
                .collect();
            let actions = match driver.handle(DriverCommand::SubmitBatch(submissions)) {
                Ok(actions) => actions,
                Err(error) => {
                    for (submission, _) in &registrations {
                        completions.remove(&submission.request.id);
                    }
                    return Err(error.into());
                }
            };
            let snapshot = driver.snapshot();
            let checkpoint = match self.persist_snapshot(&snapshot) {
                Ok(checkpoint) => checkpoint,
                Err(error) => {
                    // Err means the shared atomic operation did not publish.
                    // Visible-but-uncertain publication is not rolled back.
                    driver.restore_snapshot(prior)?;
                    for (submission, _) in &registrations {
                        completions.remove(&submission.request.id);
                    }
                    return Err(error);
                }
            };
            let terminal = registrations
                .iter()
                .filter_map(|(submission, completion)| {
                    let task_id = submission.request.id;
                    let state = snapshot.tasks.get(&task_id)?.state;
                    completion::from_task_snapshot(
                        &snapshot,
                        task_id,
                        Err(TransferFailure::Execution(format!(
                            "local task inherited terminal dependency state {state:?}"
                        ))),
                    )
                    .map(|original| (Arc::clone(completion), original))
                })
                .collect::<Vec<_>>();
            (actions, terminal, checkpoint)
        };
        drop(completions);
        for (completion, original) in inherited_terminals {
            completion.complete(original, checkpoint.clone());
        }
        Self::dispatch(Arc::clone(self), actions);
        Ok(registrations
            .into_iter()
            .map(|(_, completion)| completion)
            .collect())
    }

    fn ensure_no_graph_quarantine(&self) -> NativeExecutionResult<()> {
        if self
            .graph_quarantine
            .lock()
            .expect("Graph quarantine lock poisoned")
            .is_empty()
        {
            Ok(())
        } else {
            Err(NativeExecutionError::Protocol(
                "local worker has an unverified Graph process tree and refuses new work".into(),
            ))
        }
    }

    pub(crate) fn cancel_all(self: &Arc<Self>, reason: CancellationReason) {
        let completions = self
            .completions
            .lock()
            .expect("completion registry poisoned")
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for completion in &completions {
            if completion.graph_authority.is_none() {
                completion.cancel_with_reason(reason);
            }
        }
        let now_millis = self.clock.now_millis();
        self.apply_cancellation(DriverCommand::CancelScope {
            scope_id: self.scope_id,
            reason,
            now_millis,
        })
        .expect("native cancellation uses an admitted scope and monotonic driver clock");
        // Active Graph cancellation authority is first registered in the
        // scheduler; otherwise the child could report its verified tree
        // retirement before the scheduler knows cancellation was requested.
        for completion in &completions {
            if completion.graph_authority.is_some() {
                completion.cancel_with_reason(reason);
            }
        }
    }

    /// Cancel a single accepted task, including an attempt that raced ahead of
    /// publication of its service-level TaskHandle. Unknown task IDs were not
    /// admitted and require no scheduler cancellation.
    pub(crate) fn cancel_task(
        self: &Arc<Self>,
        task_id: TaskId,
        reason: CancellationReason,
    ) -> NativeExecutionResult<bool> {
        let completion = self
            .completions
            .lock()
            .expect("completion registry poisoned")
            .get(&task_id)
            .cloned();
        if let Some(completion) = &completion {
            if completion.graph_authority.is_none() {
                completion.cancel_with_reason(reason);
            }
        }
        let now_millis = self.clock.now_millis();
        match self.apply_cancellation(DriverCommand::CancelTask {
            task_id,
            reason,
            now_millis,
        }) {
            Ok(()) => {}
            Err(NativeExecutionError::Driver(runmat_execution_runner::RunnerError::UnknownTask(
                id,
            ))) if id == task_id => {
                self.completions
                    .lock()
                    .expect("completion registry poisoned")
                    .remove(&task_id);
                return Ok(false);
            }
            Err(error) => return Err(error),
        }
        if let Some(completion) = completion {
            if completion.graph_authority.is_some() {
                completion.cancel_with_reason(reason);
            }
        }
        Ok(true)
    }

    /// Apply one scheduler cancellation decision, persist it, and publish every
    /// task whose complete terminal history it established.
    fn apply_cancellation(
        self: &Arc<Self>,
        mut command: DriverCommand,
    ) -> NativeExecutionResult<()> {
        let completions = self
            .completions
            .lock()
            .expect("completion registry poisoned")
            .iter()
            .map(|(id, completion)| (*id, Arc::clone(completion)))
            .collect::<Vec<_>>();
        let (actions, terminal, checkpoint) = {
            let mut driver = self.driver.lock().expect("local driver poisoned");
            match &mut command {
                DriverCommand::CancelTask { now_millis, .. }
                | DriverCommand::CancelScope { now_millis, .. } => {
                    *now_millis = (*now_millis).max(driver.now_millis())
                }
                _ => unreachable!("cancellation command constructed by this adapter"),
            }
            let actions = driver.handle(command)?;
            let snapshot = driver.snapshot();
            let checkpoint = self.persist_observed_snapshot(&snapshot);
            let terminal = completions
                .into_iter()
                .filter_map(|(id, completion)| {
                    completion::from_task_snapshot(&snapshot, id, Err(TransferFailure::Cancelled))
                        .map(|original| (completion, original))
                })
                .collect::<Vec<_>>();
            (actions, terminal, checkpoint)
        };
        for (completion, original) in terminal {
            let cancelled = matches!(original.result, Err(TransferFailure::Cancelled));
            completion.complete(original, checkpoint.clone());
            if cancelled {
                // A queued cancellation must not race a later child launch.
                completion.complete_cancelled_if_no_active_attempt(checkpoint.clone());
            }
        }
        Self::dispatch(Arc::clone(self), actions);
        Ok(())
    }

    pub(crate) fn fail_collective_gang(
        &self,
        gang: &runmat_execution::GangHandle,
        reason: impl Into<String>,
    ) {
        self.collectives.fail_gang(gang, reason);
    }

    pub(crate) fn terminate_collective_rank(
        &self,
        gang: &runmat_execution::GangHandle,
        rank: runmat_types::LabRank,
    ) {
        self.collectives.rank_terminated(gang, rank);
    }

    fn dispatch(this: Arc<Self>, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => Self::launch(Arc::clone(&this), request),
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    let completion = this
                        .completions
                        .lock()
                        .expect("completion registry poisoned")
                        .get(&request.task_id)
                        .cloned();
                    if let Some(completion) = completion {
                        completion.cancel();
                    }
                }
                DriverAction::Checkpoint => {
                    this.checkpoint_observed();
                }
                DriverAction::ResizePool { .. } | DriverAction::GarbageCollectResults { .. } => {}
            }
        }
    }

    fn launch(this: Arc<Self>, request: AttemptRequest) {
        let completion = this
            .completions
            .lock()
            .expect("completion registry poisoned")
            .get(&request.task_id)
            .cloned()
            .expect("scheduled task has completion");
        if completion
            .launch_state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let started = BackendReport::for_request(&request, AttemptReport::Started);
        let actions = this
            .driver
            .lock()
            .expect("local driver poisoned")
            .handle(DriverCommand::BackendReport(started))
            .unwrap_or_default();
        Self::dispatch(Arc::clone(&this), actions);
        std::thread::spawn(move || {
            let spmd_gang = match &request.task.invocation_context {
                runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
                    Some(task.gang.clone())
                }
                _ => None,
            };
            let observed = process::execute_attempt(&this, &request, &completion);
            // Only this process owner can certify that a cancelled Graph
            // attempt has no live worker tree or foreign callback. Every
            // non-unverified result from execute_attempt has passed the
            // bounded tree-retirement path (or failed before spawn).
            let graph_cancel_registered = completion.graph_authority.is_some()
                && this
                    .driver
                    .lock()
                    .expect("local driver poisoned")
                    .snapshot()
                    .attempts
                    .get(&request.id)
                    .is_some_and(|attempt| attempt.cancellation_requested_at.is_some());
            let verified_graph_cancel = graph_cancel_registered
                && !matches!(&observed.result, Err(TransferFailure::RetirementUnverified(_)));
            let (result, report) = observed.into_backend_report(&request, verified_graph_cancel);
            let (actions, snapshot, checkpoint) = {
                let mut driver = this.driver.lock().expect("local driver poisoned");
                let actions = driver
                    .handle(DriverCommand::BackendReport(report))
                    .unwrap_or_default();
                let snapshot = driver.snapshot();
                let checkpoint = this.persist_observed_snapshot(&snapshot);
                (actions, snapshot, checkpoint)
            };
            if let Some(terminal) = completion::from_snapshot(&snapshot, &request, result) {
                let failed = terminal.result.is_err();
                // Original execution remains intact even if durable delivery fails.
                completion.complete(terminal, checkpoint.clone());
                if let (true, Some(gang)) = (failed, spmd_gang) {
                    // Publish the originating task result before releasing
                    // peers blocked in this gang's collectives.
                    this.collectives.fail_gang(&gang, "an SPMD peer failed");
                }
            }
            let _ = completion.launch_state.compare_exchange(
                1,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            if completion.is_cancelled() {
                completion.complete_cancelled_if_no_active_attempt(checkpoint);
            }
            Self::dispatch(Arc::clone(&this), actions);
        });
    }

    #[cfg(test)]
    pub(crate) fn fail_next_checkpoint_for_test(&self) {
        self.fail_next_checkpoint.store(true, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn scheduler_snapshot_for_test(&self) -> runmat_execution_runner::DriverSnapshot {
        self.driver
            .lock()
            .expect("local driver poisoned")
            .snapshot()
    }
}

impl Drop for LocalDriver {
    fn drop(&mut self) {
        let quarantine = self
            .graph_quarantine
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        process::retry_unverified_quarantine(quarantine);
        if !process::persist_unverified_quarantine(quarantine) {
            // A private scratch whose tree could not be reaped must remain
            // recoverable. Never recursively erase it as ordinary cache data.
            self.store_lifetime.retain_for_recovery();
        }
    }
}

#[cfg(test)]
mod task_completion_tests {
    use std::collections::BTreeSet;
    use std::sync::atomic::Ordering;

    use runmat_execution::resource::Capability;

    use super::completion::CompletedTask;
    use super::{LocalDriver, TaskCompletion, TransferFailure, TransferResult};
    use crate::NativeCheckpointEvidence;

    fn completed(result: TransferResult) -> CompletedTask {
        CompletedTask {
            result,
            evidence: None,
        }
    }

    #[test]
    fn completion_is_pollable_without_blocking_the_await_caller() {
        let completion = TaskCompletion::new();
        assert_eq!(completion.try_value(), None);

        let result: TransferResult = Err(TransferFailure::Execution("completed".into()));
        completion.complete(completed(result.clone()), NativeCheckpointEvidence::Durable);
        assert_eq!(completion.try_value(), Some(result));
    }

    #[test]
    fn queued_cancellation_cannot_race_a_later_child_launch() {
        let queued = TaskCompletion::new();
        queued.cancel();
        queued.complete_cancelled_if_no_active_attempt(NativeCheckpointEvidence::Durable);
        assert_eq!(queued.try_value(), Some(Err(TransferFailure::Cancelled)));
        assert!(queued
            .launch_state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err());

        let active = TaskCompletion::new();
        active.launch_state.store(1, Ordering::Release);
        active.cancel();
        active.complete_cancelled_if_no_active_attempt(NativeCheckpointEvidence::Durable);
        assert!(active.try_value().is_none());
        active.complete(
            completed(Err(TransferFailure::Cancelled)),
            NativeCheckpointEvidence::Durable,
        );
        assert_eq!(active.try_value(), Some(Err(TransferFailure::Cancelled)));
    }

    #[test]
    fn local_pool_resizes_its_schedulable_worker_inventory() {
        let temporary = tempfile::tempdir().unwrap();
        let scope = crate::config::fresh_scope_id(b"resize-test", 1);
        let driver = LocalDriver::new(
            crate::NativeExecutionConfig {
                executable: std::env::current_exe().unwrap(),
                worker_arguments: vec!["--execution-worker".into()],
                max_workers: 3,
                max_message_bytes: 1024,
                max_object_bytes: 1024,
                max_stderr_bytes: 1024,
                local_cpu_capacity_millicores: None,
                local_memory_capacity_bytes: None,
                worker_memory_bytes: 1024 * 1024 * 1024,
                worker_scratch_bytes: 1024 * 1024 * 1024,
                worker_cpu_millicores: 1000,
                store_root: temporary.path().join("session"),
                meshing_subtask_cache_root: None,
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
        .unwrap();
        assert_eq!(driver.active_workers().unwrap(), 3);
        driver.resize_pool(1).unwrap();
        assert_eq!(driver.active_workers().unwrap(), 1);
        driver.resize_pool(3).unwrap();
        assert_eq!(driver.active_workers().unwrap(), 3);
    }
}
