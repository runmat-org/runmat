use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::{Rc, Weak};

use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::state::PoolState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{
    CancellationReason, Digest, ExecutionHandleSnapshot, ExecutionHandleState, ExecutionScopeId,
    FutureHandle, FutureId, OutputContract, PoolBackend, PoolHandle, PoolId, PoolRequest,
    PoolSnapshot, TaskHandle, TaskId, TaskResultClaim,
};
use runmat_execution_artifact::{
    ProgramExecutionRequest, ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V2,
};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction,
    DriverCommand, DriverConfig, PoolSpec, TaskSubmission, WorkerSpec,
};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, ExecutionServiceError, RuntimeExecutionServices,
};
use runmat_value::Value;

use super::host::BrowserExecutionHost;
use super::model::BrowserExecutionCapabilities;
use super::resources::{
    browser_inventory, browser_request, browser_worker_inventory, driver_error,
};
use super::state::{FutureState, State, TaskRecord};

pub(crate) struct BrowserExecutionService {
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    host: Option<BrowserExecutionHost>,
    capabilities: BrowserExecutionCapabilities,
    state: RefCell<State>,
    self_weak: RefCell<Weak<Self>>,
}

impl BrowserExecutionService {
    pub(crate) fn new(
        host: Option<BrowserExecutionHost>,
    ) -> Result<Rc<Self>, ExecutionServiceError> {
        let capabilities = host
            .as_ref()
            .map(BrowserExecutionHost::capabilities)
            .unwrap_or_default();
        let session_nonce = uuid::Uuid::new_v4();
        let scope_id = ExecutionScopeId::derive(&[b"browser-session", session_nonce.as_bytes()]);
        let pool_id = PoolId::derive(&[scope_id.bytes(), b"browser"]);
        let resources = browser_inventory(capabilities);
        let mut driver = Driver::new(DriverConfig::default(), 1)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        driver
            .handle(DriverCommand::RegisterScope {
                scope_id,
                parent: None,
            })
            .map_err(driver_error)?;
        driver
            .handle(DriverCommand::CreatePool(PoolSpec {
                id: pool_id,
                min_workers: capabilities.max_workers,
                max_workers: capabilities.max_workers,
                max_in_flight: capabilities.max_workers,
                resource_limit: resources.clone(),
            }))
            .map_err(driver_error)?;
        driver
            .handle(DriverCommand::SetPoolState {
                pool_id,
                state: PoolState::Ready,
            })
            .map_err(driver_error)?;
        for index in 0..capabilities.max_workers {
            let worker_id = WorkerId::derive(&[pool_id.bytes(), &index.to_be_bytes()]);
            driver
                .handle(DriverCommand::RegisterWorker(WorkerSpec {
                    id: worker_id,
                    pool_id,
                    resources: browser_worker_inventory(capabilities),
                }))
                .map_err(driver_error)?;
        }
        let service = Rc::new(Self {
            scope_id,
            pool_id,
            host,
            capabilities,
            state: RefCell::new(State {
                next_future: 0,
                next_task: 0,
                next_completion: 0,
                futures: Default::default(),
                tasks: Default::default(),
                requests: Default::default(),
                driver,
                pool_generation: 1,
                pool_open: false,
            }),
            self_weak: RefCell::new(Weak::new()),
        });
        service.self_weak.replace(Rc::downgrade(&service));
        Ok(service)
    }

    fn validate_scope(&self, scope_id: ExecutionScopeId) -> Result<(), ExecutionServiceError> {
        if scope_id == self.scope_id {
            Ok(())
        } else {
            Err(ExecutionServiceError::ForeignScope)
        }
    }

    fn pool_backend(&self) -> PoolBackend {
        if self.capabilities.has_worker_isolation() {
            PoolBackend::BrowserWorkers
        } else {
            PoolBackend::Serial
        }
    }

    fn pool_snapshot(&self, generation: u64) -> PoolSnapshot {
        PoolSnapshot {
            handle: PoolHandle {
                id: self.pool_id,
                scope_id: self.scope_id,
                generation,
            },
            backend: self.pool_backend(),
            workers: self.capabilities.max_workers,
            state: PoolState::Ready,
        }
    }

    fn future_for_value(
        &self,
        value: Value,
    ) -> Result<Result<FutureHandle, Value>, ExecutionServiceError> {
        match value {
            Value::Future(handle) => Ok(Ok(handle)),
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.borrow();
                let task = state
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if task.generation != handle.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                Ok(Ok(FutureHandle {
                    id: task.future_id,
                    scope_id: self.scope_id,
                    outputs: handle.outputs,
                }))
            }
            value => Ok(Err(value)),
        }
    }

    fn dispatch(&self, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => self.launch(request),
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    let _ = self.cancel_attempt(&request);
                }
                DriverAction::ResizePool { .. }
                | DriverAction::Checkpoint
                | DriverAction::GarbageCollectResults { .. } => {}
            }
        }
    }

    fn launch(&self, attempt: AttemptRequest) {
        let program = {
            let mut state = self.state.borrow_mut();
            let started = BackendReport::for_request(&attempt, AttemptReport::Started);
            let actions = state
                .driver
                .handle(DriverCommand::BackendReport(started))
                .unwrap_or_default();
            drop(state);
            self.dispatch(actions);
            self.state.borrow().requests.get(&attempt.task_id).cloned()
        };
        let Some(program) = program else {
            self.finish_attempt(
                attempt,
                Err("browser scheduler lost the exact program request".into()),
            );
            return;
        };
        let weak = self.self_weak.borrow().clone();
        let host = self.host.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let response = match host {
                Some(host) => {
                    host.launch(
                        &attempt.task_id.to_string(),
                        &attempt.worker_id.to_string(),
                        &program,
                    )
                    .await
                }
                None => Ok(super::execute_local_program(program).await),
            };
            if let Some(service) = weak.upgrade() {
                service.finish_attempt(attempt, response);
            }
        });
    }

    fn finish_attempt(
        &self,
        attempt: AttemptRequest,
        response: Result<ProgramExecutionResponse, String>,
    ) {
        let (future_id, result, report) = match response {
            Ok(ProgramExecutionResponse::Success { value }) => {
                let decoded = runmat_runtime::execution::value_codec::decode_inline_value(&value)
                    .map_err(|error| ExecutionServiceError::Failed(error.to_string()));
                let report = AttemptReport::Succeeded {
                    result: AttemptSuccess {
                        outputs: vec![value],
                        result_objects: Vec::new(),
                    },
                };
                (self.future_id(attempt.task_id), decoded, report)
            }
            Ok(ProgramExecutionResponse::ExternalizedSuccess { .. }) => {
                let message =
                    "browser execution host cannot consume native externalized results".to_string();
                let result = Err(ExecutionServiceError::Failed(message.clone()));
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Rejected,
                    message,
                };
                (self.future_id(attempt.task_id), result, report)
            }
            Ok(ProgramExecutionResponse::Failure { message }) => {
                let result = Err(ExecutionServiceError::Failed(message.clone()));
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Execution,
                    message,
                };
                (self.future_id(attempt.task_id), result, report)
            }
            Err(message) => {
                let result = Err(ExecutionServiceError::Failed(message.clone()));
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Infrastructure,
                    message,
                };
                (self.future_id(attempt.task_id), result, report)
            }
        };
        let mut state = self.state.borrow_mut();
        if let Some(future_id) = future_id {
            if !matches!(state.futures.get(&future_id), Some(FutureState::Cancelled)) {
                state
                    .futures
                    .insert(future_id, FutureState::Completed(result));
                record_browser_completion(&mut state, attempt.task_id);
            }
        }
        state.requests.remove(&attempt.task_id);
        let actions = state
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &attempt, report,
            )))
            .unwrap_or_default();
        drop(state);
        self.dispatch(actions);
    }

    fn future_id(&self, task_id: TaskId) -> Option<FutureId> {
        self.state
            .borrow()
            .tasks
            .get(&task_id)
            .map(|task| task.future_id)
    }

    fn cancel_attempt(&self, attempt: &AttemptRequest) -> Result<(), ExecutionServiceError> {
        if let Some(host) = &self.host {
            host.cancel(&attempt.task_id.to_string())
                .map_err(ExecutionServiceError::Failed)?;
        }
        Ok(())
    }
}

impl RuntimeExecutionServices for BrowserExecutionService {
    fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        let state = self.state.borrow();
        Ok(state
            .pool_open
            .then(|| self.pool_snapshot(state.pool_generation)))
    }

    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        let backend = self.pool_backend();
        if request
            .backend
            .is_some_and(|requested| requested != backend)
            || request
                .workers
                .is_some_and(|workers| workers != self.capabilities.max_workers)
        {
            return Err(ExecutionServiceError::Failed(format!(
                "this browser session owns a fixed pool with {} worker(s)",
                self.capabilities.max_workers
            )));
        }
        let mut state = self.state.borrow_mut();
        state.pool_open = true;
        Ok(self.pool_snapshot(state.pool_generation))
    }

    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(pool.scope_id)?;
        let mut state = self.state.borrow_mut();
        if !state.pool_open || pool.id != self.pool_id || pool.generation != state.pool_generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        state.pool_open = false;
        state.pool_generation = state.pool_generation.wrapping_add(1);
        drop(state);
        self.drain_scope(CancellationReason::User);
        Ok(())
    }

    fn requires_program_capture(&self) -> bool {
        true
    }

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.descriptor.requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.borrow_mut();
        let sequence = state.next_future;
        state.next_future = sequence.wrapping_add(1);
        let id = FutureId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        state
            .futures
            .insert(id, FutureState::Deferred(Box::new(call)));
        Ok(FutureHandle {
            id,
            scope_id: self.scope_id,
            outputs: OutputContract { requested_outputs },
        })
    }

    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.borrow_mut();
        let call = match state.futures.get(&future.id) {
            Some(FutureState::Deferred(call)) => call.clone(),
            Some(_) => {
                return Err(ExecutionServiceError::Failed(
                    "future has already been scheduled".into(),
                ))
            }
            None => return Err(ExecutionServiceError::UnknownHandle),
        };
        let (callable, recipe, artifact, arguments) = runmat_vm::materialize_deferred_call(
            &call,
            future.outputs.clone(),
            runmat_execution_artifact::ProgramTarget::portable(
                "wasm32-browser-interpreter-bytecode-v1",
            ),
        )?;
        let sequence = state.next_task;
        state.next_task = sequence.wrapping_add(1);
        let task_id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let task_scope_id =
            ExecutionScopeId::derive(&[self.scope_id.bytes(), task_id.bytes().as_slice()]);
        state
            .driver
            .handle(DriverCommand::RegisterScope {
                scope_id: task_scope_id,
                parent: Some(self.scope_id),
            })
            .map_err(driver_error)?;
        let request = ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V2,
            recipe: recipe.clone(),
            artifact: artifact.clone(),
            callable: callable.clone(),
            arguments: arguments.clone(),
            requested_outputs: future.outputs.requested_outputs,
        };
        state.requests.insert(task_id, request);
        state
            .futures
            .insert(future.id, FutureState::Scheduled(task_id));
        let generation = 1;
        state.tasks.insert(
            task_id,
            TaskRecord {
                future_id: future.id,
                generation,
                scope_id: task_scope_id,
                read: false,
                claimed: false,
                completion_order: None,
            },
        );
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let actions = state
            .driver
            .handle(DriverCommand::Submit(Box::new(TaskSubmission {
                request: TaskRequest {
                    id: task_id,
                    scope_id: task_scope_id,
                    pool_id: self.pool_id,
                    program_artifact_id: artifact_id,
                    callable: Callable {
                        owner_identity: "browser-session".into(),
                        qualified_name: callable.display_name(),
                        entrypoint_digest: callable.identity_digest(),
                    },
                    inputs: arguments,
                    outputs: future.outputs.clone(),
                    resources: browser_request(self.capabilities),
                    retry: RetryPolicy::Never,
                    deadline_unix_millis: None,
                },
                dependencies: BTreeSet::new(),
                priority: 0,
            })))
            .map_err(driver_error)?;
        drop(state);
        self.dispatch(actions);
        Ok(TaskHandle {
            id: task_id,
            scope_id: self.scope_id,
            generation,
            outputs: future.outputs.clone(),
        })
    }

    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let state = self.state.borrow();
        let stored_state = state
            .futures
            .get(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state: browser_future_state(stored_state),
            outputs: future.outputs.clone(),
            read: state
                .tasks
                .values()
                .find(|task| task.future_id == future.id)
                .is_some_and(|task| task.read),
        })
    }

    fn inspect_task(
        &self,
        task: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.validate_scope(task.scope_id)?;
        let state = self.state.borrow();
        let record = state
            .tasks
            .get(&task.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.generation != task.generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let state = state
            .futures
            .get(&record.future_id)
            .map(browser_future_state)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state,
            outputs: task.outputs.clone(),
            read: record.read,
        })
    }

    fn claim_next_task_result(
        &self,
        tasks: &[TaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        let mut state = self.state.borrow_mut();
        let mut selected: Option<(u64, usize, TaskId)> = None;
        let mut has_unread = false;
        for (index, task) in tasks.iter().enumerate() {
            self.validate_scope(task.scope_id)?;
            let record = state
                .tasks
                .get(&task.id)
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            if record.generation != task.generation {
                return Err(ExecutionServiceError::UnknownHandle);
            }
            if record.read {
                continue;
            }
            has_unread = true;
            if record.claimed {
                continue;
            }
            if let Some(order) = record.completion_order {
                let candidate = (order, index, task.id);
                if selected.is_none_or(|current| candidate < current) {
                    selected = Some(candidate);
                }
            }
        }
        let Some((_, index, task_id)) = selected else {
            return Ok(if has_unread {
                TaskResultClaim::Pending
            } else {
                TaskResultClaim::Exhausted
            });
        };
        state
            .tasks
            .get_mut(&task_id)
            .expect("task identity was validated above")
            .claimed = true;
        Ok(TaskResultClaim::Claimed { index })
    }

    fn mark_task_result_read(&self, task: &TaskHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(task.scope_id)?;
        let mut state = self.state.borrow_mut();
        let record = state
            .tasks
            .get_mut(&task.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.generation != task.generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        record.claimed = false;
        record.read = true;
        Ok(())
    }

    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        let original = value.clone();
        let future = match self.future_for_value(value)? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.borrow_mut();
        match state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?
        {
            FutureState::Deferred(call) => {
                let call = call.clone();
                *state
                    .futures
                    .get_mut(&future.id)
                    .expect("future was just resolved") = FutureState::ExecutingInCaller;
                Ok(AwaitAction::ExecuteFuture {
                    handle: future,
                    call,
                })
            }
            FutureState::Scheduled(_) => Ok(AwaitAction::Pending(original)),
            FutureState::ExecutingInCaller => Err(ExecutionServiceError::Failed(
                "future is already executing in its caller".into(),
            )),
            FutureState::Completed(result) => {
                let result = result.clone();
                result.map(AwaitAction::Completed)
            }
            FutureState::Cancelled => Err(ExecutionServiceError::Cancelled),
        }
    }

    fn complete_future(
        &self,
        future: &FutureHandle,
        result: Result<Value, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.borrow_mut();
        let record = state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        match record {
            FutureState::ExecutingInCaller => {
                *record = FutureState::Completed(result);
                if let Some(task_id) = state
                    .tasks
                    .iter()
                    .find_map(|(task_id, task)| (task.future_id == future.id).then_some(*task_id))
                {
                    record_browser_completion(&mut state, task_id);
                }
                Ok(())
            }
            FutureState::Cancelled => Err(ExecutionServiceError::Cancelled),
            _ => Err(ExecutionServiceError::UnknownHandle),
        }
    }

    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        let (future_id, task_id, cancellation_scope) = match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.borrow();
                let scheduled_task = match state.futures.get(&handle.id) {
                    Some(FutureState::Scheduled(task_id)) => Some(*task_id),
                    _ => None,
                };
                let task_scope = scheduled_task
                    .and_then(|task_id| state.tasks.get(&task_id))
                    .map(|task| task.scope_id);
                (handle.id, scheduled_task, task_scope)
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.borrow();
                let task = state
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                (task.future_id, Some(handle.id), Some(task.scope_id))
            }
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        {
            let mut state = self.state.borrow_mut();
            state.futures.insert(future_id, FutureState::Cancelled);
            if let Some(task_id) = task_id {
                record_browser_completion(&mut state, task_id);
            }
        }
        let actions = if let Some(scope_id) = cancellation_scope {
            self.state
                .borrow_mut()
                .driver
                .handle(DriverCommand::CancelScope {
                    scope_id,
                    reason,
                    now_millis: js_sys::Date::now() as u64,
                })
                .map_err(driver_error)?
        } else {
            Vec::new()
        };
        if let Some(task_id) = task_id {
            self.state.borrow_mut().requests.remove(&task_id);
        }
        self.dispatch(actions);
        Ok(())
    }

    fn drain_scope(&self, reason: CancellationReason) {
        let actions = self
            .state
            .borrow_mut()
            .driver
            .handle(DriverCommand::CancelScope {
                scope_id: self.scope_id,
                reason,
                now_millis: js_sys::Date::now() as u64,
            })
            .unwrap_or_default();
        let mut state = self.state.borrow_mut();
        for future in state.futures.values_mut() {
            if !matches!(future, FutureState::Completed(_)) {
                *future = FutureState::Cancelled;
            }
        }
        let task_ids = state.tasks.keys().copied().collect::<Vec<_>>();
        for task_id in task_ids {
            record_browser_completion(&mut state, task_id);
        }
        state.requests.clear();
        drop(state);
        self.dispatch(actions);
    }
}

fn record_browser_completion(state: &mut State, task_id: TaskId) {
    if state
        .tasks
        .get(&task_id)
        .is_none_or(|task| task.completion_order.is_some())
    {
        return;
    }
    let order = state.next_completion;
    state.next_completion = order.wrapping_add(1);
    state
        .tasks
        .get_mut(&task_id)
        .expect("task identity was validated above")
        .completion_order = Some(order);
}

fn browser_future_state(state: &FutureState) -> ExecutionHandleState {
    match state {
        FutureState::Deferred(_) => ExecutionHandleState::Deferred,
        FutureState::ExecutingInCaller | FutureState::Scheduled(_) => ExecutionHandleState::Running,
        FutureState::Completed(Ok(_)) => ExecutionHandleState::Finished,
        FutureState::Completed(Err(_)) => ExecutionHandleState::Failed,
        FutureState::Cancelled => ExecutionHandleState::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    use super::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn deferred_test_call(program: Vec<u8>) -> DeferredCall {
        DeferredCall {
            descriptor: runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                runmat_hir::CallableIdentity::BoundFunction(runmat_hir::FunctionId(0)),
                Vec::new(),
                1,
                runmat_hir::CallableFallbackPolicy::None,
                runmat_runtime::call::descriptor::CallableCallKind::Direct,
            ),
            program_revision: None,
            program: Some(program),
        }
    }

    #[wasm_bindgen_test(async)]
    async fn serial_fallback_schedules_through_the_portable_driver() {
        let service = BrowserExecutionService::new(None).unwrap();
        assert_eq!(service.capabilities.max_workers, 1);
        assert!(!service.capabilities.has_worker_isolation());
        let future = service
            .create_future(deferred_test_call(b"invalid registry".to_vec()))
            .unwrap();
        let task = service.spawn(&future).unwrap();
        assert!(matches!(
            service.begin_await(Value::Task(task.clone())).unwrap(),
            AwaitAction::Pending(_)
        ));
        for _ in 0..8 {
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(
                &wasm_bindgen::JsValue::UNDEFINED,
            ))
            .await
            .unwrap();
            if service.begin_await(Value::Task(task.clone())).is_err() {
                return;
            }
        }
        panic!("serial browser execution did not report its worker failure");
    }

    #[wasm_bindgen_test]
    fn cancelling_a_scheduled_future_cancels_its_child_task() {
        let service = BrowserExecutionService::new(None).unwrap();
        let future = service
            .create_future(deferred_test_call(b"invalid registry".to_vec()))
            .unwrap();
        let task = service.spawn(&future).unwrap();
        service
            .cancel(
                &Value::Future(future),
                runmat_execution::CancellationReason::User,
            )
            .unwrap();
        assert_eq!(
            service.begin_await(Value::Task(task)),
            Err(ExecutionServiceError::Cancelled)
        );
    }

    #[wasm_bindgen_test]
    fn browser_pool_lifecycle_reports_exact_topology_and_fences_stale_handles() {
        let service = BrowserExecutionService::new(None).unwrap();
        let first = service.current_pool().unwrap().expect("browser pool");
        assert_eq!(first.backend, PoolBackend::Serial);
        assert_eq!(first.workers, 1);
        service.close_pool(&first.handle).unwrap();
        assert!(service.current_pool().unwrap().is_none());
        let reopened = service.ensure_pool(PoolRequest::automatic()).unwrap();
        assert_ne!(first.handle.generation, reopened.handle.generation);
    }
}
