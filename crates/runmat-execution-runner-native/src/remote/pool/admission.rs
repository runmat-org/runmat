//! Atomic original task/graph admission and exact idempotent replay.
use super::*;

struct PendingTaskRegistration {
    task_id: TaskId,
    program: ProgramExecutionRequest,
    submission: TaskSubmission,
    completion_receiver: oneshot::Receiver<(u64, CompletionResult)>,
    progress: Arc<super::super::pool_progress::RemoteProgressBuffer>,
}

struct ExistingTask {
    task_id: TaskId,
    committed: Option<AttemptSuccess>,
}

impl RemotePoolDriver {
    /// Admit the existing Direct or selected-Graph route without changing the
    /// caller's task grant. Graph requires an authenticated staged transfer.
    pub fn submit_worker(
        self: &Arc<Self>,
        program: WorkerProgramRequest,
        submission: TaskSubmission,
        transfer: Option<RemoteGraphTransfer>,
    ) -> NativeExecutionResult<RemoteTaskCompletion> {
        match program {
            WorkerProgramRequest::Direct(program) => {
                if transfer.is_some() {
                    return Err(NativeExecutionError::Protocol(
                        "Direct program cannot claim a Graph transfer".into(),
                    ));
                }
                self.submit(submission, *program)
            }
            WorkerProgramRequest::Graph(graph) => {
                graph
                    .validate()
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
                let task = &submission.request;
                let expected =
                    runmat_execution::identity::ArtifactId::derive(&[graph.artifact.id.0.bytes()]);
                let callable = graph
                    .scheduler_callable()
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
                let host = graph
                    .execution_host_requirement(task.host.permitted_trust_tiers.clone())
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
                let schema = runmat_execution::value::GraphRequiredSchema::from_version(
                    graph.schema_version,
                )
                .ok_or_else(|| NativeExecutionError::Protocol("unsupported Graph schema".into()))?;
                if task.program_artifact_id() != Some(expected)
                    || task.callable.program != callable
                    || task.host != host
                    || task.outputs != graph.recipe.outputs
                    || task.invocation_context != graph.context
                    || !runmat_execution_artifact::GraphInvocationControls::matches_original(
                        graph.invocation_controls.as_ref(),
                        &task.environment,
                        task.console,
                    )
                    || !task.inputs.is_empty()
                    || graph.assignment.is_some()
                    || graph.job_id.is_some()
                    || !task
                        .resources
                        .required_capabilities
                        .contains(&runmat_execution::resource::Capability::ProcessIsolation)
                    || schema.required_capability().is_some_and(|capability| {
                        !task.resources.required_capabilities.contains(&capability)
                    })
                {
                    return Err(NativeExecutionError::Protocol(
                        "remote Graph task differs from its exact selected invocation".into(),
                    ));
                }
                let transfer = transfer.ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "Graph program requires an authenticated transfer".into(),
                    )
                })?;
                let (digest, length) = crate::metadata_identity::json_identity(
                    b"",
                    &graph,
                    transfer.maximum_program_bytes,
                )?;
                if digest != transfer.program_digest || length != transfer.program_length {
                    return Err(NativeExecutionError::Protocol(
                        "Graph program changed after encrypted transfer admission".into(),
                    ));
                }
                self.submit_graph_program(
                    submission,
                    *graph,
                    usize::try_from(length).map_err(|_| {
                        NativeExecutionError::Protocol(
                            "Graph length exceeds host address space".into(),
                        )
                    })?,
                    Some(transfer),
                )
            }
        }
    }

    pub fn submit(
        self: &Arc<Self>,
        submission: TaskSubmission,
        program: ProgramExecutionRequest,
    ) -> NativeExecutionResult<RemoteTaskCompletion> {
        self.submit_graph(vec![(program, submission)])?
            .pop()
            .ok_or_else(|| {
                NativeExecutionError::Protocol("remote task admission returned no handle".into())
            })
    }

    fn existing_task(
        &self,
        submission: &TaskSubmission,
        program: &ProgramExecutionRequest,
    ) -> NativeExecutionResult<Option<ExistingTask>> {
        self.input_inventories
            .lock()
            .expect("input inventory registry poisoned")
            .validate(submission.request.id, program)?;
        self.validate_direct_task(submission, program)?;
        let existing = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .tasks
            .get(&submission.request.id)
            .cloned();
        let Some(existing) = existing else {
            return Ok(None);
        };
        // Retry, deadline, and priority govern an attempt that has already finished; they do not
        // change the committed program semantics. Everything capable of changing the result or
        // its admissible placement must still agree before the commit can be reused.
        if existing.submission.request.id != submission.request.id
            || existing.submission.request.scope_id != submission.request.scope_id
            || existing.submission.request.pool_id != submission.request.pool_id
            || existing.submission.request.executable != submission.request.executable
            || existing.submission.request.callable != submission.request.callable
            || existing.submission.request.host != submission.request.host
            || existing.submission.request.invocation_context
                != submission.request.invocation_context
            || existing.submission.request.environment != submission.request.environment
            || existing.submission.request.console != submission.request.console
            || existing.submission.request.inputs != submission.request.inputs
            || existing.submission.request.outputs != submission.request.outputs
            || existing.submission.request.resources != submission.request.resources
            || existing.submission.dependencies != submission.dependencies
        {
            return Err(NativeExecutionError::Protocol(format!(
                "remote task {} collides with a different immutable submission",
                submission.request.id
            )));
        }
        if existing.state != runmat_execution::state::TaskState::Succeeded {
            if self
                .restored_completions
                .lock()
                .expect("restored completion registry poisoned")
                .contains_key(&submission.request.id)
            {
                return Ok(Some(ExistingTask {
                    task_id: submission.request.id,
                    committed: None,
                }));
            }
            return Err(NativeExecutionError::Protocol(format!(
                "remote task {} already exists in non-reusable state {:?}",
                submission.request.id, existing.state
            )));
        }
        let committed = existing.committed.ok_or_else(|| {
            NativeExecutionError::Protocol(format!(
                "succeeded remote task {} has no committed result",
                submission.request.id
            ))
        })?;
        for reference in committed.result.result_objects() {
            if self.execution_object(reference)?.is_none() {
                return Err(NativeExecutionError::Protocol(format!(
                    "committed remote task {} has an unavailable result object",
                    submission.request.id
                )));
            }
        }
        Ok(Some(ExistingTask {
            task_id: submission.request.id,
            committed: Some(committed.result),
        }))
    }

    /// Direct V7 tasks run only against this pool's installed bundle and must
    /// carry exactly the admitted program's identity, invocation and inputs.
    fn validate_direct_task(
        &self,
        submission: &TaskSubmission,
        program: &ProgramExecutionRequest,
    ) -> NativeExecutionResult<()> {
        if self.direct_bundle.is_none() {
            return Err(NativeExecutionError::Configuration(
                "direct request cannot use a graph-only remote pool".into(),
            ));
        }
        super::super::direct_admission::validate(
            program,
            super::super::direct_admission::Admission::Pool {
                task: &submission.request,
                scope_id: self.scope_id,
                pool_id: self.pool_id,
            },
        )
        .map_err(|message| NativeExecutionError::Protocol(message.into()))
    }

    /// Atomically admits a topologically ordered graph before dispatching any task.
    pub fn submit_graph(
        self: &Arc<Self>,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
    ) -> NativeExecutionResult<Vec<RemoteTaskCompletion>> {
        if tasks.is_empty() {
            return Err(NativeExecutionError::Protocol(
                "remote task graph must not be empty".into(),
            ));
        }
        // Existing restored handles are consumed only after the whole graph validates
        // and any new tasks are admitted. Failed graph submission cannot steal a waiter.
        let _admission = self.admission.lock().expect("remote admission poisoned");
        let mut ordered = Vec::with_capacity(tasks.len());
        let mut pending = Vec::new();
        let mut unique = BTreeSet::new();
        for (program, submission) in tasks {
            if !unique.insert(submission.request.id) {
                return Err(NativeExecutionError::Protocol(
                    "remote graph repeats a task".into(),
                ));
            }
            if let Some(existing) = self.existing_task(&submission, &program)? {
                ordered.push(Some(existing));
            } else {
                ordered.push(None);
                pending.push((program, submission));
            }
        }
        let mut admitted = if pending.is_empty() {
            Vec::new()
        } else {
            self.submit_new_graph(pending)?
        }
        .into_iter();
        let mut restored = self
            .restored_completions
            .lock()
            .expect("restored completion registry poisoned");
        ordered
            .into_iter()
            .map(|existing| {
                let completion = match existing {
                    Some(ExistingTask {
                        task_id,
                        committed: Some(result),
                    }) => {
                        let wall = self
                            .task_wall_intervals
                            .lock()
                            .expect("remote task wall intervals poisoned")
                            .get(&task_id)
                            .and_then(|interval| match interval {
                                super::timing::TaskWallInterval::Completed(wall) => Some(*wall),
                                super::timing::TaskWallInterval::Started(_) => None,
                                super::timing::TaskWallInterval::ObservedPending(_) => None,
                            });
                        restored.remove(&task_id);
                        Some(match wall {
                            Some(wall) => {
                                RemoteTaskCompletion::resolved_committed(task_id, result, wall)
                            }
                            None if self
                                .task_clocks
                                .lock()
                                .expect("task clock bindings poisoned")
                                .contains_key(&task_id) =>
                            {
                                RemoteTaskCompletion::resolved_pending(task_id, result)
                            }
                            None => {
                                return Err(NativeExecutionError::Protocol(
                                    "committed replay lacks its original host wall interval".into(),
                                ))
                            }
                        })
                    }
                    Some(ExistingTask {
                        task_id,
                        committed: None,
                    }) => restored
                        .remove(&task_id)
                        .map(|completion| completion.resume_existing(task_id)),
                    None => admitted.next(),
                };
                completion.ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "remote task graph omitted an admitted completion".into(),
                    )
                })
            })
            .collect()
    }

    fn submit_new_graph(
        self: &Arc<Self>,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
    ) -> NativeExecutionResult<Vec<RemoteTaskCompletion>> {
        // One physical host clock begins before scheduler admission and worker dispatch.
        let started = std::time::Instant::now();
        let mut registrations = Vec::with_capacity(tasks.len());
        let mut completion_senders = Vec::with_capacity(tasks.len());
        for (program, submission) in tasks {
            self.validate_direct_task(&submission, &program)?;
            let (sender, receiver) = oneshot::channel();
            let task_id = submission.request.id;
            registrations.push(PendingTaskRegistration {
                task_id,
                program,
                submission,
                completion_receiver: receiver,
                progress: Arc::new(super::super::pool_progress::RemoteProgressBuffer::default()),
            });
            completion_senders.push((task_id, sender));
        }
        let mut inventories = self
            .input_inventories
            .lock()
            .expect("input inventory registry poisoned");
        let mut admitted_inventories = inventories.clone();
        for registration in &registrations {
            admitted_inventories.bind(registration.task_id, &registration.program)?;
        }
        let mut programs = self
            .programs
            .lock()
            .expect("remote program catalog poisoned");
        let mut completions = self
            .completions
            .lock()
            .expect("remote completion registry poisoned");
        let mut progress_registry = self
            .progress
            .lock()
            .expect("remote progress registry poisoned");
        let mut unique = BTreeSet::new();
        if let Some(registration) = registrations.iter().find(|registration| {
            let task_id = registration.task_id;
            !unique.insert(task_id)
                || programs.contains_key(&task_id)
                || completions.contains_key(&task_id)
                || progress_registry.contains_key(&task_id)
        }) {
            return Err(NativeExecutionError::Protocol(format!(
                "remote task {} already has registered execution state",
                registration.task_id
            )));
        }
        for registration in &registrations {
            programs.insert(
                registration.task_id,
                Arc::new(WorkerProgramRequest::direct(registration.program.clone())),
            );
            progress_registry.insert(registration.task_id, Arc::clone(&registration.progress));
        }
        for (task_id, sender) in completion_senders {
            completions.insert(task_id, sender);
        }
        let submissions = registrations
            .iter()
            .map(|registration| registration.submission.clone())
            .collect();
        let mut driver = self.driver.lock().expect("remote driver poisoned");
        let actions = match driver.handle(DriverCommand::SubmitBatch(submissions)) {
            Ok(actions) => actions,
            Err(error) => {
                for registration in &registrations {
                    programs.remove(&registration.task_id);
                    completions.remove(&registration.task_id);
                    progress_registry.remove(&registration.task_id);
                }
                return Err(error.into());
            }
        };
        {
            let mut intervals = self
                .task_wall_intervals
                .lock()
                .expect("remote task wall intervals poisoned");
            for registration in &registrations {
                let observed = self
                    .task_clocks
                    .lock()
                    .expect("task clock bindings poisoned")
                    .get(&registration.task_id)
                    .copied();
                match (observed, intervals.get(&registration.task_id)) {
                    (
                        Some(owner),
                        Some(super::timing::TaskWallInterval::ObservedPending(existing)),
                    ) if owner == *existing => {}
                    (None, None) => {
                        intervals.insert(
                            registration.task_id,
                            super::timing::TaskWallInterval::Started(started),
                        );
                    }
                    _ => {
                        return Err(NativeExecutionError::Protocol(
                            "remote task wall interval differs from admission authority".into(),
                        ));
                    }
                }
            }
        }
        drop(driver);
        drop(progress_registry);
        drop(completions);
        drop(programs);
        *inventories = admitted_inventories;
        drop(inventories);
        self.dispatch(actions);
        self.resolve_non_success_terminals();
        Ok(registrations
            .into_iter()
            .map(|registration| {
                RemoteTaskCompletion::new(registration.completion_receiver, registration.progress)
            })
            .collect())
    }

    /// Admit one V10 Graph Program to a graph-only pool. Its signed staged
    /// transfer is retained for retries; workers receive only its digest.
    pub(in crate::remote) fn submit_graph_program(
        self: &Arc<Self>,
        submission: TaskSubmission,
        graph: GraphExecutionRequest,
        encoded_bytes: usize,
        transfer: Option<SignedGraphTransfer>,
    ) -> NativeExecutionResult<RemoteTaskCompletion> {
        if self.direct_bundle.is_some() {
            return Err(NativeExecutionError::Configuration(
                "graph request cannot use a direct-bundle remote pool".into(),
            ));
        }
        if submission.request.scope_id != self.scope_id
            || submission.request.pool_id != self.pool_id
        {
            return Err(NativeExecutionError::Configuration(
                "remote task is outside this driver authority".into(),
            ));
        }
        graph.validate().map_err(|error| {
            NativeExecutionError::Protocol(format!("remote graph request is invalid: {error}"))
        })?;
        graph
            .recipe
            .validate_resource_request(&submission.request.resources)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let demand = graph
            .input_allocation_floor_bytes(encoded_bytes)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        if submission.request.resources.memory_bytes < demand
            || !submission.request.inputs.is_empty()
            || submission.request.outputs.requested_outputs != graph.requested_outputs
            || submission.request.invocation_context != graph.context
        {
            return Err(NativeExecutionError::Protocol(
                "remote graph workload does not match its resource or invocation admission".into(),
            ));
        }
        let transfer = transfer.ok_or_else(|| {
            NativeExecutionError::Protocol(
                "remote Graph execution has no authenticated staged transfer".into(),
            )
        })?;
        if transfer.program_length != encoded_bytes as u64 {
            return Err(NativeExecutionError::Protocol(
                "remote Graph allocation length differs from its staged Program".into(),
            ));
        }
        let _admission = self.admission.lock().expect("remote admission poisoned");
        let task_id = submission.request.id;
        let (sender, receiver) = oneshot::channel();
        let progress = Arc::new(super::super::pool_progress::RemoteProgressBuffer::default());
        let mut programs = self
            .programs
            .lock()
            .expect("remote program catalog poisoned");
        let mut completions = self
            .completions
            .lock()
            .expect("remote completion registry poisoned");
        let mut progress_registry = self
            .progress
            .lock()
            .expect("remote progress registry poisoned");
        if programs.contains_key(&task_id)
            || completions.contains_key(&task_id)
            || progress_registry.contains_key(&task_id)
        {
            return Err(NativeExecutionError::Protocol(format!(
                "remote task {task_id} already has registered execution state"
            )));
        }
        let actions = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .handle(DriverCommand::Submit(Box::new(submission)))?;
        programs.insert(task_id, Arc::new(WorkerProgramRequest::graph(graph)));
        self.graph_transfers
            .lock()
            .expect("remote Graph transfer registry poisoned")
            .insert(task_id, Arc::new(transfer));
        completions.insert(task_id, sender);
        progress_registry.insert(task_id, Arc::clone(&progress));
        drop(progress_registry);
        drop(completions);
        drop(programs);
        self.dispatch(actions);
        self.resolve_non_success_terminals();
        Ok(RemoteTaskCompletion::new(receiver, progress))
    }
}
