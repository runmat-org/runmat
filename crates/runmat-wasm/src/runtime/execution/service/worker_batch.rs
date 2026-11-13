//! Real isolated browser worker waves. Publication uses the existing driver;
//! a oneshot terminal event, not a task-turn polling loop, wakes its owner.
use super::*;
use futures::channel::oneshot;
use futures::future::{LocalBoxFuture, Shared};
use futures::FutureExt;
mod physical;
use super::program_preparation::PreparedProgram;
pub(in crate::runtime::execution) use physical::PhysicalCallback;
use runmat_runtime::execution::DeferredProgram;

pub(super) struct TaskWake {
    sender: Option<oneshot::Sender<()>>,
    pub(super) receiver: Shared<LocalBoxFuture<'static, Result<(), ExecutionServiceError>>>,
}
impl TaskWake {
    pub(super) fn new() -> Self {
        let (sender, receiver) = oneshot::channel();
        Self {
            sender: Some(sender),
            receiver: async move {
                receiver.await.map_err(|_| {
                    ExecutionServiceError::Infrastructure(
                        "task completion owner ended before physical worker settlement".into(),
                    )
                })
            }
            .boxed_local()
            .shared(),
        }
    }
}
impl BrowserExecutionService {
    pub(super) fn release_callbacks(
        &self,
        tasks: &[TaskHandle],
    ) -> Result<(), ExecutionServiceError> {
        use runmat_runtime::execution::TaskRetention;
        {
            let mut state = self.state.borrow_mut();
            for task in tasks {
                self.validate_scope(task.scope_id)?;
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
                if let Some(record) = state.tasks.get_mut(&task.id) {
                    record.retention = TaskRetention::ReleasingCallbackBatch;
                }
            }
        }
        for task in tasks {
            self.retire_released_callback(task.id);
        }
        Ok(())
    }
    pub(super) fn retire_released_callback(&self, id: TaskId) {
        use runmat_runtime::execution::TaskRetention;
        let retired = {
            let mut state = self.state.borrow_mut();
            let Some(record) = state
                .tasks
                .get(&id)
                .filter(|t| t.retention == TaskRetention::ReleasingCallbackBatch)
            else {
                return;
            };
            let future = record.future_id;
            if record
                .callback_exit
                .as_ref()
                .is_some_and(PhysicalCallback::needs_retirement)
                || state
                    .driver
                    .task_record(id)
                    .is_ok_and(|task| task.active_attempt.is_some())
            {
                return;
            }
            if !matches!(
                state.futures.get(&future),
                Some(
                    FutureState::Cancelled
                        | FutureState::Completed(_)
                        | FutureState::GraphCompleted { .. }
                        | FutureState::GraphFailed { .. }
                        | FutureState::GraphResolved { .. }
                )
            ) {
                return;
            }
            let retired = state.futures.remove(&future);
            state.tasks.remove(&id);
            state.requests.remove(&id);
            retired
        };
        drop(retired);
        self.wake_callback(id);
    }
    pub(super) fn wake_callback(&self, id: TaskId) {
        // Never wake a caller while holding service/driver RefCell borrows.
        let signal = self.callback_waiters.borrow_mut().remove(&id);
        if let Some(mut signal) = signal {
            if let Some(sender) = signal.sender.take() {
                let _ = sender.send(());
            }
        }
    }
    pub(super) fn settle_callback_cancellations(&self) {
        let ids = self
            .callback_waiters
            .borrow()
            .keys()
            .copied()
            .collect::<Vec<_>>();
        let mut retired = Vec::new();
        let mut completed = Vec::new();
        {
            let mut state = self.state.borrow_mut();
            for id in ids {
                let Some(future) = state.tasks.get(&id).map(|t| t.future_id) else {
                    continue;
                };
                if matches!(
                    state.futures.get(&future),
                    Some(FutureState::CancellingCallback { .. })
                ) && state
                    .driver
                    .task_record(id)
                    .is_ok_and(|task| task.attempt_count == 0 && task.active_attempt.is_none())
                {
                    retired.push(state.futures.insert(future, FutureState::Cancelled));
                    state.requests.remove(&id);
                    record_browser_completion(&mut state, id);
                }
                if state
                    .tasks
                    .get(&id)
                    .is_some_and(|task| task.completion_order.is_some())
                {
                    completed.push(id);
                }
            }
        }
        drop(retired);
        for id in completed {
            self.wake_callback(id);
            self.retire_released_callback(id);
        }
    }

    pub(super) fn submit_selected_callbacks(
        &self,
        calls: Vec<DeferredCall>,
        pool: &PoolHandle,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<Vec<TaskHandle>, ExecutionServiceError> {
        if cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        if !self.supports_callback_batches() {
            return Err(ExecutionServiceError::Infrastructure(
                "browser host has no isolated settled-worker contract".into(),
            ));
        }
        if !self.callback_workers_healthy() {
            return Err(ExecutionServiceError::WorkerLost(
                "selected callback worker retirement is unverified".into(),
            ));
        }
        let current = self.inspect_pool(pool)?;
        if current.backend != PoolBackend::BrowserWorkers || calls.len() > current.workers as usize
        {
            return Err(ExecutionServiceError::Failed(
                "callback batch exceeds selected browser pool".into(),
            ));
        }
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
            service: self,
            futures: Vec::with_capacity(calls.len()),
            complete: false,
        };
        {
            let state = self.state.borrow();
            state
                .next_future
                .checked_add(calls.len() as u64)
                .ok_or_else(|| {
                    ExecutionServiceError::Infrastructure(
                        "browser future identity exhausted".into(),
                    )
                })?;
        }
        for call in calls {
            created.futures.push(self.create_future(call)?);
        }
        let futures = created.futures.as_slice();
        let mut reservations = Vec::with_capacity(futures.len());
        let mut prepared = Vec::with_capacity(futures.len());
        for future in futures {
            if cancellation.is_cancelled() {
                return Err(ExecutionServiceError::Cancelled);
            }
            let (id, call) = self.reserve_worker_materialization(future)?;
            reservations.push(MaterializationGuard {
                service: self,
                future_id: future.id,
                id,
                armed: true,
            });
            let request =
                PreparedProgram::prepare(&call, future.outputs.clone(), self.capabilities)?;
            request.admit_registered_worker(self)?;
            prepared.push((request, call.retry));
        }
        if cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        self.advance_driver_now()?;
        let mut tasks = Vec::with_capacity(futures.len());
        let actions;
        let mut retired = Vec::with_capacity(futures.len());
        {
            let mut state = self.state.borrow_mut();
            for reservation in &reservations {
                if !matches!(state.futures.get(&reservation.future_id),Some(FutureState::Materializing{id,cancelled:false,..}) if *id==reservation.id)
                {
                    return Err(ExecutionServiceError::Cancelled);
                }
            }
            let mut submissions = Vec::with_capacity(futures.len());
            let mut publications = Vec::with_capacity(futures.len());
            for (future, (prepared, retry)) in futures.iter().zip(prepared) {
                let sequence = state.next_task;
                state.next_task = sequence.checked_add(1).ok_or_else(|| {
                    ExecutionServiceError::Infrastructure("browser task identity exhausted".into())
                })?;
                let id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
                let scope_id =
                    ExecutionScopeId::derive(&[self.scope_id.bytes(), id.bytes().as_slice()]);
                state
                    .driver
                    .handle(DriverCommand::RegisterScope {
                        scope_id,
                        parent: Some(self.scope_id),
                    })
                    .map_err(driver_error)?;
                let (submission, request, receipt) = prepared.into_submission(
                    id,
                    scope_id,
                    self.pool_id,
                    future.outputs.clone(),
                    retry,
                );
                submissions.push(submission);
                tasks.push(TaskHandle {
                    id,
                    scope_id: self.scope_id,
                    generation: 1,
                    outputs: future.outputs.clone(),
                });
                publications.push((id, scope_id, request, receipt));
            }
            // This command has existing all-or-none task publication. No
            // host launch occurs until every request and receipt is installed.
            actions = state
                .driver
                .handle(DriverCommand::SubmitBatch(submissions))
                .map_err(driver_error)?;
            for ((future, (id, scope_id, request, receipt)), reservation) in
                futures.iter().zip(publications).zip(&mut reservations)
            {
                state.requests.insert(id, request);
                state.tasks.insert(
                    id,
                    TaskRecord {
                        future_id: future.id,
                        generation: 1,
                        scope_id,
                        read: false,
                        claimed: false,
                        completion_order: None,
                        retention: runmat_runtime::execution::TaskRetention::CallbackBatch,
                        callback_exit: Some(PhysicalCallback::default()),
                    },
                );
                retired.push(state.futures.insert(
                    future.id,
                    FutureState::Scheduled {
                        task_id: id,
                        receipt,
                    },
                ));
                self.callback_waiters
                    .borrow_mut()
                    .insert(id, TaskWake::new());
                reservation.armed = false;
            }
        }
        created.complete = true;
        drop(retired);
        self.dispatch(actions);
        Ok(tasks)
    }

    pub(super) fn reserve_worker_materialization(
        &self,
        future: &FutureHandle,
    ) -> Result<(u64, Rc<DeferredCall>), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.borrow_mut();
        let call = match state.futures.get(&future.id) {
            Some(FutureState::Deferred(call)) => Rc::clone(call),
            Some(_) => {
                return Err(ExecutionServiceError::Failed(
                    "future has already been scheduled".into(),
                ))
            }
            None => return Err(ExecutionServiceError::UnknownHandle),
        };
        call.require_spawn_admission(&future.outputs)?;
        let id = state.next_materialization;
        state.next_materialization = id.checked_add(1).ok_or_else(|| {
            ExecutionServiceError::Infrastructure(
                "browser materialization identity exhausted".into(),
            )
        })?;
        let retired = state.futures.insert(
            future.id,
            FutureState::Materializing {
                id,
                call: Rc::clone(&call),
                cancelled: false,
            },
        );
        drop(state);
        drop(retired);
        Ok((id, call))
    }
}
struct CreatedFutures<'a> {
    service: &'a BrowserExecutionService,
    futures: Vec<FutureHandle>,
    complete: bool,
}
impl Drop for CreatedFutures<'_> {
    fn drop(&mut self) {
        if !self.complete {
            for future in &self.futures {
                let _ = self.service.cancel(
                    &Value::Future(future.clone().into()),
                    CancellationReason::ParentScope,
                );
            }
            let retired = {
                let mut state = self.service.state.borrow_mut();
                self.futures
                    .iter()
                    .filter_map(|future| {
                        let accepted = state.tasks.values().any(|task| task.future_id == future.id);
                        if !accepted
                            && matches!(state.futures.get(&future.id), Some(FutureState::Cancelled))
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
