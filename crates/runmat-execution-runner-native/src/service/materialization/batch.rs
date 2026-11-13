//! Atomic selected-Graph callback waves using existing reservations/driver.
use super::*;
use crate::driver::LocalWorkerSubmission;
use runmat_execution::PoolHandle;
use runmat_runtime::execution::RuntimeExecutionServices;

pub(in crate::service) fn release(
    service: &NativeExecutionService,
    tasks: &[TaskHandle],
) -> Result<(), ExecutionServiceError> {
    use runmat_runtime::execution::TaskRetention;
    let mut retired = Vec::with_capacity(tasks.len());
    let mut released = Vec::with_capacity(tasks.len());
    {
        let mut state = service.state.lock().expect("native service poisoned");
        // Validate the whole exact owner list before changing any record.
        for task in tasks {
            service.validate_scope(task.scope_id)?;
            if let Some(record) = state.tasks.get(&task.id) {
                if record.generation != task.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                if record.retention == TaskRetention::Refetchable {
                    return Err(ExecutionServiceError::Failed(
                        "public Future is not an internal callback result".into(),
                    ));
                }
            }
        }
        for task in tasks {
            let Some(record) = state.tasks.get_mut(&task.id) else {
                continue;
            };
            record.retention = TaskRetention::ReleasingCallbackBatch;
            let future = record.future_id;
            let verified = record
                .callback_completion
                .as_ref()
                .is_some_and(|completion| completion.verified_terminal() == Some(true));
            let terminal = match state.futures.get(&future) {
                Some(
                    FutureState::Cancelled
                    | FutureState::Completed(_)
                    | FutureState::GraphResolved { .. },
                ) => true,
                Some(FutureState::CancellingGraph(running)) => {
                    running.completion().verified_terminal() == Some(true)
                }
                _ => false,
            };
            if terminal && verified {
                retired.push(state.futures.remove(&future));
                state.tasks.remove(&task.id);
                released.push(task.id);
            } else if !state.cancelling_graphs.contains(&future) {
                state.cancelling_graphs.push(future);
            }
        }
    }
    // Rooted result clones already handed to the caller retain their own
    // root/source owners through decode; foreign-resource drops may reenter.
    drop(retired);
    for task in released {
        service.driver.release_completed_callback(task);
    }
    Ok(())
}

pub(in crate::service) fn submit(
    service: &NativeExecutionService,
    calls: Vec<DeferredCall>,
    pool: &PoolHandle,
    cancellation: runmat_execution::cancellation::CancellationToken,
    owner: Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
) -> Result<Vec<TaskHandle>, ExecutionServiceError> {
    if cancellation.is_cancelled() {
        return Err(ExecutionServiceError::Cancelled);
    }
    let current = service.inspect_pool(pool)?;
    if current.backend != runmat_execution::PoolBackend::LocalProcesses
        || calls.len() > current.workers as usize
    {
        return Err(ExecutionServiceError::Failed(
            "callback batch exceeds selected native pool".into(),
        ));
    }
    // All shape checks precede insertion; generic/Direct callbacks continue
    // through their existing owner instead of receiving a fake Graph receipt.
    for call in &calls {
        if !matches!(
            call.program,
            DeferredProgram::SelectedGraph(_) | DeferredProgram::InterpreterV2(_)
        ) || call.invocation.requested_outputs() > u16::MAX as usize
        {
            return Err(ExecutionServiceError::InvalidOutputContract);
        }
    }
    let mut created = CreatedFutures {
        service,
        futures: Vec::with_capacity(calls.len()),
        complete: false,
    };
    for call in calls {
        created.futures.push(service.create_future(call)?);
    }
    let futures = created.futures.as_slice();
    // Each exact reservation restores/cancels its original deferred owner on
    // materialization failure or unwinding. No state lock spans a snapshot.
    let mut guards = Vec::with_capacity(futures.len());
    let mut selected = Vec::with_capacity(futures.len());
    for future in futures {
        if cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        let (id, call) = reserve(&service.state, future)?;
        guards.push(MaterializationGuard {
            state: &service.state,
            driver: &service.driver,
            future,
            id,
            task_id: None,
            armed: true,
            submitting: false,
        });
        let prepared = prepare_submission(&call, future.outputs.clone())?;
        selected.push((call, prepared));
    }
    let mut ids = Vec::with_capacity(futures.len());
    {
        let mut state = service.state.lock().expect("native service poisoned");
        for guard in &guards {
            if state
                .futures
                .get(&guard.future.id)
                .and_then(|s| same_reservation(s, guard.id))
                != Some(false)
            {
                return Err(ExecutionServiceError::Cancelled);
            }
        }
        for _ in futures {
            let sequence = state.next_task;
            state.next_task = sequence.checked_add(1).ok_or_else(|| {
                ExecutionServiceError::Infrastructure("native task identity exhausted".into())
            })?;
            ids.push(TaskId::derive(&[
                service.scope_id.bytes(),
                &sequence.to_be_bytes(),
            ]));
        }
    }
    let mut submissions = Vec::with_capacity(futures.len());
    let mut receipts = Vec::with_capacity(futures.len());
    for (((call, prepared), id), guard) in selected.into_iter().zip(&ids).zip(&mut guards) {
        guard.task_id = Some(*id);
        guard.submitting = true;
        match prepared {
            PreparedSubmission::Graph(prepared) => {
                submissions.push(LocalWorkerSubmission::Graph(Box::new(
                    LocalGraphSubmission {
                        task_id: *id,
                        request: prepared.request,
                        retry: call.retry,
                        task: None,
                    },
                )));
                receipts.push(Some(prepared.receipt));
            }
            PreparedSubmission::Direct(prepared) => {
                let PreparedDirectSubmission {
                    callable,
                    invocation_context,
                    recipe,
                    artifact,
                    inputs,
                } = *prepared;
                submissions.push(LocalWorkerSubmission::Direct(Box::new(
                    LocalProgramSubmission {
                        task_id: *id,
                        callable,
                        invocation_context,
                        recipe,
                        artifact,
                        inputs,
                        outputs: guard.future.outputs.clone(),
                        retry: call.retry,
                        environment: call.environment.clone(),
                        console: call.console,
                    },
                )));
                receipts.push(None);
            }
        }
    }
    if cancellation.is_cancelled() {
        return Err(ExecutionServiceError::Cancelled);
    }
    let completions = service
        .driver
        .submit_local_workers_with_control(submissions, owner)
        .map_err(|error| ExecutionServiceError::Infrastructure(error.to_string()))?;
    // The canonical atomic publisher returns one completion for each submitted
    // task. There is no fallible handoff between publication and installation.
    // Publication is atomic; observers can now see only this whole admitted
    // wave. No GC/source operation occurs while installing completion owners.
    let mut retired = Vec::with_capacity(futures.len());
    let mut tasks = Vec::with_capacity(futures.len());
    let mut cancelling = Vec::new();
    {
        let mut state = service.state.lock().expect("native service poisoned");
        let wave_cancelled = cancellation.is_cancelled()
            || guards.iter().any(|guard| {
                state
                    .futures
                    .get(&guard.future.id)
                    .and_then(|s| same_reservation(s, guard.id))
                    != Some(false)
            });
        for ((((future, id), completion), receipt), guard) in futures
            .iter()
            .zip(ids)
            .zip(completions)
            .zip(receipts)
            .zip(&mut guards)
        {
            let running = RunningCompletion::new(Arc::clone(&completion), receipt);
            // If any reservation was cancelled, no handle is returned for the
            // wave: every accepted task must therefore enter owned retirement.
            let cancelled = wave_cancelled;
            let next = if cancelled {
                state.cancelling_graphs.push(future.id);
                cancelling.push(id);
                FutureState::CancellingGraph(running)
            } else {
                FutureState::Running(running)
            };
            retired.push(state.futures.insert(future.id, next));
            state.tasks.insert(
                id,
                TaskRecord {
                    future_id: future.id,
                    generation: 1,
                    read: false,
                    claimed: false,
                    completion_order: None,
                    retention: if cancelled {
                        runmat_runtime::execution::TaskRetention::ReleasingCallbackBatch
                    } else {
                        runmat_runtime::execution::TaskRetention::CallbackBatch
                    },
                    callback_completion: Some(completion),
                },
            );
            guard.armed = false;
            tasks.push(TaskHandle {
                id,
                scope_id: service.scope_id,
                generation: 1,
                outputs: future.outputs.clone(),
            });
        }
    }
    drop(retired);
    created.complete = true;
    if !cancelling.is_empty() {
        for id in cancelling {
            let _ = service
                .driver
                .cancel_task(id, CancellationReason::ParentScope);
        }
        service.sweep_completed_graph_cancellations();
        return Err(ExecutionServiceError::Cancelled);
    }
    Ok(tasks)
}

struct CreatedFutures<'a> {
    service: &'a NativeExecutionService,
    futures: Vec<FutureHandle>,
    complete: bool,
}
impl Drop for CreatedFutures<'_> {
    fn drop(&mut self) {
        if !self.complete {
            for future in &self.futures {
                let _ = self.service.cancel(
                    &runmat_value::Value::Future(future.clone().into()),
                    CancellationReason::ParentScope,
                );
            }
            // These handles were minted only by this private producer. An
            // unpublished terminal slot has no refetch API. Materialization
            // guards can settle a failed submission as Completed(Err), rather
            // than Cancelled; cancelling a terminal future preserves that
            // original failure, so cancellation alone cannot retire its slot.
            let retired = {
                let mut state = self.service.state.lock().expect("native service poisoned");
                self.futures
                    .iter()
                    .filter_map(|future| {
                        let accepted = state.tasks.values().any(|task| task.future_id == future.id);
                        if !accepted
                            && matches!(
                                state.futures.get(&future.id),
                                Some(FutureState::Cancelled | FutureState::Completed(_))
                            )
                        {
                            state.futures.remove(&future.id)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            };
            drop(retired);
        }
    }
}
