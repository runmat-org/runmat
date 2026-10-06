use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::ValuePayload;
use runmat_execution::{
    CancellationReason, Digest, ExecutionScopeId, OutputContract, PoolId, TaskId,
};
use runmat_execution_artifact::{ProgramArtifact, ProgramBuildRecipe};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptRequest, Driver, DriverAction, DriverCommand, DriverConfig,
    TaskSubmission,
};

mod process;

use crate::local_store::{prepare_session_root, ArtifactStore, CheckpointStore};
use crate::protocol::StoredProgram;
use crate::{
    FilesystemObjectStore, NativeExecutionConfig, NativeExecutionError, NativeExecutionResult,
};

use crate::process_provider::TransferResult;

mod clock;
mod completion;
mod persistence;
pub(crate) use completion::TaskCompletion;

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
    completions: Mutex<HashMap<TaskId, Arc<TaskCompletion>>>,
}

impl LocalDriver {
    pub(crate) fn new(
        config: NativeExecutionConfig,
        scope_id: ExecutionScopeId,
    ) -> NativeExecutionResult<Arc<Self>> {
        let mut driver = Driver::new(DriverConfig::default(), 1)?;
        let pool_id = crate::native_pool::register(&mut driver, &config, scope_id)?;
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
            completions: Mutex::new(HashMap::new()),
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

    pub(crate) fn submit(
        self: &Arc<Self>,
        task_id: TaskId,
        function: usize,
        recipe: ProgramBuildRecipe,
        artifact: ProgramArtifact,
        inputs: Vec<ValuePayload>,
        outputs: OutputContract,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let request = TaskRequest {
            id: task_id,
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            program_artifact_id: artifact_id,
            callable: Callable {
                owner_identity: "local-session".into(),
                qualified_name: function.to_string(),
                entrypoint_digest: Digest::sha256(function.to_be_bytes()),
            },
            inputs,
            outputs,
            resources: ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes: 1024 * 1024,
                scratch_bytes: 1024 * 1024,
                max_wall_millis: 24 * 60 * 60 * 1000,
                max_artifact_bytes: u64::from(self.config.max_message_bytes),
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            },
            retry: RetryPolicy::Never,
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
            function,
        )
    }

    pub(crate) fn submit_task(
        self: &Arc<Self>,
        submission: TaskSubmission,
        recipe: ProgramBuildRecipe,
        artifact: ProgramArtifact,
        function: usize,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        self.submit_tasks(vec![(submission, recipe, artifact, function)])?
            .pop()
            .ok_or_else(|| {
                NativeExecutionError::Protocol("local task admission returned no completion".into())
            })
    }

    pub(crate) fn submit_tasks(
        self: &Arc<Self>,
        tasks: Vec<(TaskSubmission, ProgramBuildRecipe, ProgramArtifact, usize)>,
    ) -> NativeExecutionResult<Vec<Arc<TaskCompletion>>> {
        if tasks.is_empty() {
            return Err(NativeExecutionError::Protocol(
                "local task submission batch must not be empty".into(),
            ));
        }
        let mut registrations = Vec::with_capacity(tasks.len());
        for (submission, recipe, artifact, function) in tasks {
            artifact.validate_against(&recipe).map_err(|error| {
                NativeExecutionError::Protocol(format!(
                    "local program artifact failed validation: {error}"
                ))
            })?;
            let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
            if submission.request.scope_id != self.scope_id
                || submission.request.pool_id != self.pool_id
                || submission.request.program_artifact_id != artifact_id
            {
                return Err(NativeExecutionError::Protocol(
                    "local task submission differs from its session or program artifact".into(),
                ));
            }
            let stored = serde_json::to_vec(&StoredProgram {
                recipe,
                artifact,
                function,
            })
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            self.artifacts.put(artifact_id, &stored)?;
            registrations.push((
                submission.request.id,
                Arc::new(TaskCompletion::new()),
                submission,
            ));
        }

        let mut completions = self
            .completions
            .lock()
            .expect("completion registry poisoned");
        let mut unique = BTreeSet::new();
        if registrations
            .iter()
            .any(|(task_id, _, _)| !unique.insert(*task_id) || completions.contains_key(task_id))
        {
            return Err(NativeExecutionError::Protocol(
                "local task batch contains a duplicate or registered task".into(),
            ));
        }
        for (task_id, completion, _) in &registrations {
            completions.insert(*task_id, Arc::clone(completion));
        }
        let (actions, inherited_terminals, checkpoint) = {
            let mut driver = self.driver.lock().expect("local driver poisoned");
            let prior = driver.snapshot();
            let submissions = registrations
                .iter()
                .map(|(_, _, submission)| submission.clone())
                .collect();
            let actions = match driver.handle(DriverCommand::SubmitBatch(submissions)) {
                Ok(actions) => actions,
                Err(error) => {
                    for (task_id, _, _) in &registrations {
                        completions.remove(task_id);
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
                    for (task_id, _, _) in &registrations {
                        completions.remove(task_id);
                    }
                    return Err(error);
                }
            };
            let terminal = registrations
                .iter()
                .filter_map(|(task_id, completion, _)| {
                    let state = snapshot.tasks.get(task_id)?.state;
                    completion::from_task_snapshot(
                        &snapshot,
                        *task_id,
                        Err(format!(
                            "local task inherited terminal dependency state {state:?}"
                        )),
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
            .map(|(_, completion, _)| completion)
            .collect())
    }

    pub(crate) fn cancel_task(
        self: &Arc<Self>,
        task_id: TaskId,
        reason: CancellationReason,
    ) -> NativeExecutionResult<()> {
        let now_millis = self.clock.now_millis();
        self.apply_cancellation(DriverCommand::CancelTask {
            task_id,
            reason,
            now_millis,
        })
    }

    pub(crate) fn cancel_all(self: &Arc<Self>, reason: CancellationReason) {
        let now_millis = self.clock.now_millis();
        self.apply_cancellation(DriverCommand::CancelScope {
            scope_id: self.scope_id,
            reason,
            now_millis,
        })
        .expect("native cancellation uses an admitted scope and monotonic driver clock");
    }

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
                    completion::from_task_snapshot(
                        &snapshot,
                        id,
                        Err("original cancellation decision".into()),
                    )
                    .map(|original| (completion, original))
                })
                .collect::<Vec<_>>();
            (actions, terminal, checkpoint)
        };
        for (completion, original) in terminal {
            completion.complete(original, checkpoint.clone());
        }
        Self::dispatch(Arc::clone(self), actions);
        Ok(())
    }

    fn dispatch(this: Arc<Self>, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => Self::launch(Arc::clone(&this), request),
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    if let Some(completion) = this
                        .completions
                        .lock()
                        .expect("completion registry poisoned")
                        .get(&request.task_id)
                    {
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
        let started = BackendReport::for_request(&request, AttemptReport::Started);
        let actions = this
            .driver
            .lock()
            .expect("local driver poisoned")
            .handle(DriverCommand::BackendReport(started))
            .expect("native start report binds an assigned attempt");
        Self::dispatch(Arc::clone(&this), actions);
        std::thread::spawn(move || {
            let completion = this
                .completions
                .lock()
                .expect("completion registry poisoned")
                .get(&request.task_id)
                .cloned()
                .expect("scheduled task has completion");
            let observed = process::execute_attempt(&this, &request, &completion);
            let (result, report) = observed.into_report(&request, completion.as_ref());
            let (actions, snapshot, checkpoint) = {
                let mut driver = this.driver.lock().expect("local driver poisoned");
                let actions = driver
                    .handle(DriverCommand::BackendReport(report))
                    .expect("native provider report binds its original attempt observation");
                let snapshot = driver.snapshot();
                let checkpoint = this.persist_observed_snapshot(&snapshot);
                (actions, snapshot, checkpoint)
            };
            if let Some(terminal) = completion::from_snapshot(&snapshot, &request, result) {
                // Original execution remains intact even if durable delivery fails.
                completion.complete(terminal, checkpoint);
            }
            Self::dispatch(Arc::clone(&this), actions);
        });
    }
}
