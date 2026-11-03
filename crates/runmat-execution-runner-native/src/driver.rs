use std::collections::{BTreeSet, HashMap};
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::ValuePayload;
use runmat_execution::{
    CancellationReason, ExecutionScopeId, OutputContract, PoolId, ProgramCallable, TaskId,
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
pub(crate) mod process;

mod clock;
mod completion;
mod console;
mod persistence;
pub(crate) use completion::{next_task_completion_order, GraphAttemptAuthority, TaskCompletion};

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

pub(crate) enum LocalWorkerSubmission {
    Direct(Box<LocalProgramSubmission>),
    Graph(Box<LocalGraphSubmission>),
}

pub(crate) struct LocalGraphSubmission {
    pub(crate) task_id: TaskId,
    pub(crate) request: GraphExecutionRequest,
    pub(crate) retry: RetryPolicy,
    pub(crate) task: Option<TaskSubmission>,
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
    console_admission: Mutex<console::LocalConsoleAuthority>,
    parent_console_attempts: Mutex<BTreeSet<runmat_execution::identity::AttemptId>>,
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
            console_admission: Mutex::new(console::LocalConsoleAuthority::StandaloneOriginalTask),
            parent_console_attempts: Mutex::new(BTreeSet::new()),
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

    pub(crate) fn submit_with_control(
        self: &Arc<Self>,
        submission: LocalProgramSubmission,
        owner: Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        self.ensure_no_graph_quarantine()?;
        let (task, recipe, artifact) = self.local_program_submission(submission)?;
        let registration = self.prepare_task(task, recipe, artifact)?;
        self.publish_controlled_registration(registration, owner)
    }

    fn publish_controlled_registration(
        self: &Arc<Self>,
        registration: (TaskSubmission, Arc<TaskCompletion>),
        owner: Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        let registration = Self::with_control_owner(registration, owner)?;
        self.publish_submissions(vec![registration])?
            .pop()
            .ok_or_else(|| {
                NativeExecutionError::Protocol("controlled admission returned no completion".into())
            })
    }

    fn with_control_owner(
        mut registration: (TaskSubmission, Arc<TaskCompletion>),
        owner: Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
    ) -> NativeExecutionResult<(TaskSubmission, Arc<TaskCompletion>)> {
        Arc::get_mut(&mut registration.1)
            .ok_or_else(|| {
                NativeExecutionError::Protocol("unpublished completion was already shared".into())
            })?
            .install_raw_control_owner(owner)?;
        Ok(registration)
    }

    fn local_program_submission(
        &self,
        submission: LocalProgramSubmission,
    ) -> NativeExecutionResult<(TaskSubmission, ProgramBuildRecipe, ProgramArtifact)> {
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
        let context_bytes =
            runmat_execution::console::invocation_memory_bytes(&environment, console)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let memory_bytes = (1024_u64 * 1024)
            .checked_add(context_bytes)
            .ok_or_else(|| {
                NativeExecutionError::Protocol("invocation memory reservation overflow".into())
            })?;
        let memory_bytes = memory_bytes
            .checked_add(
                crate::fea_proxy::automatic_storage_bytes()
                    .map_err(NativeExecutionError::Protocol)?,
            )
            .ok_or_else(|| {
                NativeExecutionError::Protocol("local FEA control reservation overflow".into())
            })?;
        let console_budget = self.quote_console_subgrant(console, memory_bytes)?;
        let memory_bytes =
            crate::console_retention::allocation_floor(memory_bytes, console, console_budget)?;
        let max_artifact_bytes = crate::console_retention::artifact_floor(
            u64::from(self.config.max_message_bytes),
            console,
            console_budget,
        )?;
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let host = artifact
            .execution_host_requirement(
                &recipe,
                BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
            )
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let accelerator_requirements = recipe.accelerators.clone();
        let scratch_bytes =
            crate::process_output::RawOutputBudget::local_scratch_bytes(1024 * 1024, console)?;
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
                scratch_bytes,
                max_wall_millis: 24 * 60 * 60 * 1000,
                max_artifact_bytes,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: accelerator_requirements,
                required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            },
            retry,
            deadline_unix_millis: None,
        };
        Ok((
            TaskSubmission {
                request,
                dependencies: BTreeSet::new(),
                priority: 0,
            },
            recipe,
            artifact,
        ))
    }

    /// Atomically admits a topologically ordered batch before dispatching any task.
    pub(crate) fn submit_tasks(
        self: &Arc<Self>,
        tasks: Vec<(TaskSubmission, ProgramBuildRecipe, ProgramArtifact)>,
    ) -> NativeExecutionResult<Vec<Arc<TaskCompletion>>> {
        self.ensure_no_graph_quarantine()?;
        let mut registrations = Vec::with_capacity(tasks.len());
        for (submission, recipe, artifact) in tasks {
            registrations.push(self.prepare_task(submission, recipe, artifact)?);
        }
        self.publish_submissions(registrations)
    }

    fn prepare_task(
        &self,
        submission: TaskSubmission,
        recipe: ProgramBuildRecipe,
        artifact: ProgramArtifact,
    ) -> NativeExecutionResult<(TaskSubmission, Arc<TaskCompletion>)> {
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
        Ok((submission, Arc::new(TaskCompletion::new())))
    }

    /// Same staged registration and atomic publication owner as submit_workers,
    /// without building a redundant request/argument copy for local producers.
    pub(crate) fn submit_local_workers_with_control(
        self: &Arc<Self>,
        tasks: Vec<LocalWorkerSubmission>,
        owner: Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
    ) -> NativeExecutionResult<Vec<Arc<TaskCompletion>>> {
        self.ensure_no_graph_quarantine()?;
        let mut prepared = Vec::with_capacity(tasks.len());
        for task in tasks {
            let registration = match task {
                LocalWorkerSubmission::Direct(task) => {
                    let (task, recipe, artifact) = self.local_program_submission(*task)?;
                    self.prepare_task(task, recipe, artifact)?
                }
                LocalWorkerSubmission::Graph(graph) => self.prepare_graph(*graph)?,
            };
            prepared.push(Self::with_control_owner(registration, owner.clone())?);
        }
        self.publish_submissions(prepared)
    }

    pub(crate) fn submit_workers(
        self: &Arc<Self>,
        tasks: Vec<(
            runmat_execution_artifact::WorkerProgramRequest,
            TaskSubmission,
        )>,
    ) -> NativeExecutionResult<Vec<Arc<TaskCompletion>>> {
        self.ensure_no_graph_quarantine()?;
        let mut prepared = Vec::with_capacity(tasks.len());
        for (request, submission) in tasks {
            prepared.push(match request {
                runmat_execution_artifact::WorkerProgramRequest::Direct(request) => {
                    self.prepare_task(submission, request.recipe, request.artifact)?
                }
                runmat_execution_artifact::WorkerProgramRequest::Graph(request) => self
                    .prepare_graph(LocalGraphSubmission {
                        task_id: submission.request.id,
                        retry: submission.request.retry,
                        request: *request,
                        task: Some(submission),
                    })?,
            });
        }
        self.publish_submissions(prepared)
    }

    pub(crate) fn submit_graph_with_control(
        self: &Arc<Self>,
        graph: LocalGraphSubmission,
        owner: Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        self.ensure_no_graph_quarantine()?;
        let registration = self.prepare_graph(graph)?;
        self.publish_controlled_registration(registration, owner)
    }

    fn prepare_graph(
        &self,
        graph: LocalGraphSubmission,
    ) -> NativeExecutionResult<(TaskSubmission, Arc<TaskCompletion>)> {
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
        if let Some(original) = &graph.task {
            crate::graph_submission::validate_original(selected, original)?;
            if original.request.id != graph.task_id
                || original.request.scope_id != self.scope_id
                || original.request.pool_id != self.pool_id
                || original.request.retry != graph.retry
            {
                return Err(NativeExecutionError::Protocol(
                    "Graph task differs from its original local execution context or retry policy"
                        .into(),
                ));
            }
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
        let memory_bytes = if graph.task.is_none() {
            memory_bytes
                .checked_add(
                    crate::fea_proxy::automatic_storage_bytes()
                        .map_err(NativeExecutionError::Protocol)?,
                )
                .ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "local Graph FEA control reservation overflow".into(),
                    )
                })?
        } else {
            memory_bytes
        };
        let materialized_scratch_bytes =
            crate::attempt_scratch::materialized_bytes(admitted.selected_bundles())?;
        let recipe = &selected.recipe;
        let artifact_id = stored.id;
        let response_authority = admitted
            .response_authority()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let invocation_context = selected.context.clone();
        let invocation_controls = selected.invocation_controls.clone().unwrap_or_default();
        let console_budget = self.quote_console_subgrant(
            invocation_controls.console,
            graph
                .task
                .as_ref()
                .map_or(memory_bytes, |task| task.request.resources.memory_bytes),
        )?;
        let memory_bytes = crate::console_retention::allocation_floor(
            memory_bytes,
            invocation_controls.console,
            console_budget,
        )?;
        let max_artifact_bytes = crate::console_retention::artifact_floor(
            stored.length,
            invocation_controls.console,
            console_budget,
        )?;
        let scratch_bytes = crate::process_output::RawOutputBudget::local_scratch_bytes(
            stored
                .length
                .max(1024 * 1024)
                .max(materialized_scratch_bytes),
            invocation_controls.console,
        )?;
        let requested_outputs = selected.requested_outputs;
        let required_capabilities =
            crate::graph_submission::required_capabilities(selected.schema_version)?;
        let submission = if let Some(original) = graph.task {
            crate::graph_submission::validate_floor(selected, &original, stored.length)?;
            crate::graph_submission::validate_scratch(&admitted, &original)?;
            original
        } else {
            TaskSubmission {
                request: TaskRequest {
                    id: graph.task_id,
                    scope_id: self.scope_id,
                    pool_id: self.pool_id,
                    schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                    executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                        artifact_id,
                    ),
                    callable: Callable::for_program("local-session", &callable),
                    host,
                    invocation_context: invocation_context.clone(),
                    inputs: Vec::new(),
                    environment: invocation_controls.environment.clone(),
                    console: invocation_controls.console,
                    outputs: recipe.outputs.clone(),
                    resources: ResourceRequest {
                        cpu_millicores: 1000,
                        memory_bytes,
                        scratch_bytes,
                        max_wall_millis: self
                            .remote_graph_max_wall_millis
                            .unwrap_or(24 * 60 * 60 * 1000),
                        max_artifact_bytes,
                        max_egress_bytes: 0,
                        max_relay_bytes: 0,
                        accelerators: recipe.accelerators.clone(),
                        required_capabilities,
                    },
                    retry: graph.retry,
                    deadline_unix_millis: None,
                },
                dependencies: BTreeSet::new(),
                priority: 0,
            }
        };
        let request = &submission.request;
        if console_budget.is_some()
            && (memory_bytes > request.resources.memory_bytes
                || max_artifact_bytes > request.resources.max_artifact_bytes)
        {
            return Err(NativeExecutionError::Protocol(
                "Graph console subtree and original inputs exceed the original task reservation"
                    .into(),
            ));
        }
        crate::process_output::RawOutputBudget::from_grant(
            request.resources.scratch_bytes,
            materialized_scratch_bytes,
            request.console,
        )?;
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
        if self
            .remote_graph_max_wall_millis
            .is_some_and(|maximum| request.resources.max_wall_millis > maximum)
        {
            return Err(NativeExecutionError::Protocol(
                "Graph wall-time ceiling exceeds its authenticated remote allocation".into(),
            ));
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
        let completion = Arc::new(TaskCompletion::new_graph(GraphAttemptAuthority {
            artifact_id: stored.id,
            stored_digest: stored.digest,
            stored_length: stored.length,
            request_artifact_id: request.program_artifact_id().ok_or_else(|| {
                NativeExecutionError::Protocol(
                    "Graph task lacks its authenticated source artifact identity".into(),
                )
            })?,
            context: invocation_context,
            requested_outputs,
            callable,
            response: response_authority,
            invocation_controls,
            materialized_scratch_bytes,
        }));
        Ok((submission, completion))
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

    /// The service's internal-result retention owner calls this only after
    /// decode or verified cancellation. Scheduler/checkpoint history remains;
    /// the redundant live completion/response source-map cache does not.
    #[cfg(all(test, unix))]
    pub(crate) fn settled_callback_completion_for_test() -> Arc<TaskCompletion> {
        let completion = Arc::new(TaskCompletion::new());
        completion.complete(
            completion::CompletedTask {
                result: Err(TransferFailure::Cancelled),
                evidence: None,
            },
            crate::NativeCheckpointEvidence::Durable,
        );
        completion
    }

    pub(crate) fn release_completed_callback(&self, task_id: TaskId) {
        let mut completions = self
            .completions
            .lock()
            .expect("completion registry poisoned");
        let retired = if completions
            .get(&task_id)
            .is_some_and(|c| c.verified_terminal() == Some(true))
        {
            completions.remove(&task_id)
        } else {
            None
        };
        drop(completions);
        drop(retired);
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
            Err(NativeExecutionError::Driver(
                runmat_execution_runner::RunnerError::UnknownTask(id),
            )) if id == task_id => {
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
                        let cause = this
                            .driver
                            .lock()
                            .expect("local driver poisoned")
                            .task_record(request.task_id)
                            .ok()
                            .and_then(|task| task.cancellation.map(|c| c.cause));
                        let selected = cause
                            .ok_or_else(|| {
                                "scheduler stop has no original cancellation cause".to_string()
                            })
                            .and_then(|cause| {
                                crate::process_provider::cancellation::reason(&request, cause)
                            });
                        match selected {
                            Ok(reason) => completion.cancel_with_reason(reason),
                            Err(message) => completion.fail_control(message),
                        }
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
            // Only this process owner's private proof can certify no live
            // worker tree or foreign callback. Neither an ordinary error nor
            // the public root-exit observation establishes that certificate.
            let cancel_registered = this
                .driver
                .lock()
                .expect("local driver poisoned")
                .snapshot_ref()
                .attempts
                .get(&request.id)
                .is_some_and(|attempt| attempt.cancellation_requested_at.is_some());
            let retired = observed.has_retirement_proof();
            let verified_cancel = retired
                && cancel_registered
                && matches!(&observed.result, Err(TransferFailure::Cancelled));
            let (result, report) = observed.into_backend_report(&request, verified_cancel);
            let mut retirement_report =
                BackendReport::for_request(&request, AttemptReport::Retired);
            retirement_report.unsuccessful_resources = report
                .unsuccessful_resources
                .clone()
                .or_else(|| match &report.report {
                    AttemptReport::Succeeded { result } => result
                        .resources()
                        .filter(|resources| {
                            resources.validate_against(&request).is_ok()
                                && report.execution.as_ref().is_none_or(|execution| {
                                    resources.resources.usage.wall_millis == execution.wall_millis
                                })
                        })
                        .cloned()
                        .map(Box::new),
                    _ => None,
                });
            retirement_report.execution = report.execution.clone();
            retirement_report.console = report.console.clone();
            let (actions, snapshot, checkpoint) = {
                let mut driver = this.driver.lock().expect("local driver poisoned");
                let mut actions = driver
                    .handle(DriverCommand::BackendReport(report))
                    .unwrap_or_default();
                if retired
                    && driver
                        .snapshot_ref()
                        .attempts
                        .get(&request.id)
                        .is_some_and(|attempt| {
                            attempt
                                .unverified_retirement
                                .as_ref()
                                .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none())
                        })
                {
                    // Preserve the original operation failure above; only this
                    // physical owner can now settle its retained reservation.
                    actions.extend(
                        driver
                            .handle(DriverCommand::BackendReport(retirement_report))
                            .unwrap_or_default(),
                    );
                }
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
            let _ =
                completion
                    .launch_state
                    .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire);
            if completion.is_cancelled() {
                completion.complete_cancelled_if_no_active_attempt(checkpoint);
            }
            Self::dispatch(Arc::clone(&this), actions);
        });
    }

    #[cfg(all(test, unix))]
    pub(crate) fn fail_next_checkpoint_for_test(&self) {
        self.fail_next_checkpoint.store(true, Ordering::Release);
    }

    #[cfg(all(test, unix))]
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
                local_fea_organization: Default::default(),
                executable: std::env::current_exe().unwrap(),
                worker_arguments: vec!["--execution-worker".into()],
                max_workers: 3,
                max_message_bytes: 1024,
                max_object_bytes: 1024,
                max_stderr_bytes: 1024,
                local_cpu_capacity_millicores: None,
                local_memory_capacity_bytes: None,
                authored_worker_memory_bytes: None,
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
