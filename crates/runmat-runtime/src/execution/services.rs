use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionHandleState, ExecutionScopeId,
    FutureHandle, FutureId, JobHandle, OutputContract, PoolBackend, PoolHandle, PoolId,
    PoolRequest, PoolSnapshot, PoolState, TaskHandle, TaskId, TaskResultClaim,
};
use runmat_value::Value;

use super::ExecutionServiceError;

static NEXT_SERVICE_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq)]
pub enum DeferredInvocation {
    Callable(crate::call::descriptor::CallableDescriptor),
    Program {
        callable: runmat_execution::ProgramCallable,
        context: runmat_execution::ProgramInvocationContext,
        arguments: Vec<Value>,
        requested_outputs: usize,
    },
}

impl DeferredInvocation {
    pub fn requested_outputs(&self) -> usize {
        match self {
            Self::Callable(descriptor) => descriptor.requested_outputs,
            Self::Program {
                requested_outputs, ..
            } => *requested_outputs,
        }
    }

    pub fn arguments(&self) -> &[Value] {
        match self {
            Self::Callable(descriptor) => &descriptor.args,
            Self::Program { arguments, .. } => arguments,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeferredCall {
    pub invocation: DeferredInvocation,
    /// Retry semantics proven at the call's semantic origin. Execution
    /// adapters must carry this value unchanged into their task request.
    pub retry: runmat_execution::RetryPolicy,
    pub program_revision: Option<runmat_execution::ProgramRevision>,
    /// Exact, runtime-opaque program description supplied by the VM.
    ///
    /// The serial service does not inspect this payload. Execution adapters may
    /// require it to reproduce the callable in an isolated worker.
    pub program: Option<Vec<u8>>,
}

/// Exact compiler-bound SPMD program submitted as one gang operation.
/// Execution backends may place ranks in separate workers, but may not split
/// admission, retry, or result identity into unrelated ordinary futures.
#[derive(Clone, Debug, PartialEq)]
pub struct SpmdGangCall {
    pub gang: runmat_execution::GangHandle,
    pub region: runmat_types::ParallelRegionId,
    pub captures: Vec<Value>,
    pub requested_outputs: usize,
    pub program_revision: Option<runmat_execution::ProgramRevision>,
    pub program: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpmdRankResult {
    pub rank: runmat_types::LabRank,
    pub outputs: Vec<Option<runmat_execution::value::ValuePayload>>,
}

impl SpmdGangCall {
    pub fn validate(&self) -> Result<(), ExecutionServiceError> {
        self.gang
            .validate()
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        if self.requested_outputs > u16::MAX as usize
            || self.captures.len() > 4096
            || self.program.is_empty()
        {
            return Err(ExecutionServiceError::Failed(
                "SPMD gang call is empty or exceeds its bounded execution contract".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DurableJobOptions {
    pub idempotency_key: Option<String>,
    pub retention_millis: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AwaitAction {
    Passthrough(Value),
    /// The host has started the operation but cannot complete it synchronously.
    ///
    /// The VM yields once and polls `begin_await` again with this value. This
    /// keeps browser workers non-blocking while native adapters may continue
    /// to use an efficient blocking completion primitive.
    Pending(Value),
    ExecuteFuture {
        handle: FutureHandle,
        call: Box<DeferredCall>,
    },
    Completed(Value),
}

pub trait RuntimeExecutionServices {
    fn scope_id(&self) -> ExecutionScopeId;
    fn requires_program_capture(&self) -> bool {
        false
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError>;
    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError>;
    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError>;
    fn execute_spmd_gang(
        &self,
        _call: SpmdGangCall,
    ) -> crate::context::RuntimeServiceFuture<Result<Vec<SpmdRankResult>, ExecutionServiceError>>
    {
        Box::pin(async {
            Err(ExecutionServiceError::Failed(
                "this execution backend does not provide isolated SPMD gangs".into(),
            ))
        })
    }
    fn inspect_pool(&self, pool: &PoolHandle) -> Result<PoolSnapshot, ExecutionServiceError> {
        let snapshot = self
            .current_pool()?
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if snapshot.handle == *pool {
            Ok(snapshot)
        } else if snapshot.handle.scope_id == pool.scope_id {
            Err(ExecutionServiceError::UnknownHandle)
        } else {
            Err(ExecutionServiceError::ForeignScope)
        }
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError>;
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError>;
    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError>;
    fn inspect_task(
        &self,
        task: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError>;
    fn claim_next_task_result(
        &self,
        tasks: &[TaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError>;
    fn mark_task_result_read(&self, task: &TaskHandle) -> Result<(), ExecutionServiceError>;
    fn spawn_on(
        &self,
        future: &FutureHandle,
        pool: Option<&PoolHandle>,
    ) -> Result<TaskHandle, ExecutionServiceError> {
        if let Some(pool) = pool {
            let current = self
                .current_pool()?
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            if current.handle != *pool {
                return Err(if current.handle.scope_id == pool.scope_id {
                    ExecutionServiceError::UnknownHandle
                } else {
                    ExecutionServiceError::ForeignScope
                });
            }
        }
        self.spawn(future)
    }
    fn submit_job(
        &self,
        _call: DeferredCall,
        _options: DurableJobOptions,
    ) -> Result<JobHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::Failed(
            "durable jobs are unavailable in this execution backend".into(),
        ))
    }
    fn await_job(&self, _job: &JobHandle) -> Result<Value, ExecutionServiceError> {
        Err(ExecutionServiceError::Failed(
            "durable jobs are unavailable in this execution backend".into(),
        ))
    }
    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError>;
    fn complete_future(
        &self,
        future: &FutureHandle,
        result: Result<Value, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError>;
    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError>;
    fn drain_scope(&self, reason: CancellationReason);
}

#[derive(Clone, Debug)]
enum FutureState {
    Deferred(Box<DeferredCall>),
    Running,
    Completed(Result<Value, ExecutionServiceError>),
    Cancelled,
}

#[derive(Clone, Debug)]
struct TaskRecord {
    future_id: FutureId,
    generation: u64,
    cancelled: bool,
    read: bool,
    claimed: bool,
    completion_order: Option<u64>,
}

#[derive(Debug)]
struct ServiceState {
    next_future: u64,
    next_task: u64,
    next_completion: u64,
    futures: HashMap<FutureId, FutureState>,
    tasks: HashMap<TaskId, TaskRecord>,
    pool_generation: u64,
    pool_open: bool,
}

/// Root-scoped serial execution service.
///
/// It is the correctness backend used when no process/worker scheduler is
/// composed. Handles and state transitions are real; execution placement is
/// deliberately delegated to later scheduler adapters.
#[derive(Debug)]
pub struct RuntimeExecutionService {
    scope_id: ExecutionScopeId,
    state: Mutex<ServiceState>,
}

impl RuntimeExecutionService {
    pub fn new() -> Self {
        let nonce = NEXT_SERVICE_NONCE.fetch_add(1, Ordering::Relaxed);
        Self {
            scope_id: ExecutionScopeId::derive(&[&nonce.to_be_bytes()]),
            state: Mutex::new(ServiceState {
                next_future: 0,
                next_task: 0,
                next_completion: 0,
                futures: HashMap::new(),
                tasks: HashMap::new(),
                pool_generation: 1,
                pool_open: false,
            }),
        }
    }

    fn validate_scope(&self, scope_id: ExecutionScopeId) -> Result<(), ExecutionServiceError> {
        if scope_id == self.scope_id {
            Ok(())
        } else {
            Err(ExecutionServiceError::ForeignScope)
        }
    }

    fn pool_snapshot(&self, generation: u64) -> PoolSnapshot {
        PoolSnapshot {
            handle: PoolHandle {
                id: PoolId::derive(&[self.scope_id.bytes(), b"serial"]),
                scope_id: self.scope_id,
                generation,
            },
            backend: PoolBackend::Serial,
            workers: 1,
            state: PoolState::Ready,
        }
    }
}

impl Default for RuntimeExecutionService {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeExecutionServices for RuntimeExecutionService {
    fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        let state = self.state.lock().expect("execution service state poisoned");
        Ok(state
            .pool_open
            .then(|| self.pool_snapshot(state.pool_generation)))
    }

    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        if request.workers.is_some_and(|workers| workers != 1)
            || request
                .backend
                .is_some_and(|backend| backend != PoolBackend::Serial)
        {
            return Err(ExecutionServiceError::Failed(
                "the serial execution backend provides exactly one worker".into(),
            ));
        }
        let mut state = self.state.lock().expect("execution service state poisoned");
        state.pool_open = true;
        Ok(self.pool_snapshot(state.pool_generation))
    }

    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(pool.scope_id)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        if !state.pool_open || pool.generation != state.pool_generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        state.pool_open = false;
        state.pool_generation = state.pool_generation.wrapping_add(1);
        drop(state);
        self.drain_scope(CancellationReason::User);
        Ok(())
    }

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.invocation.requested_outputs())
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        let sequence = state.next_future;
        state.next_future = state.next_future.wrapping_add(1);
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
        let mut state = self.state.lock().expect("execution service state poisoned");
        match state.futures.get(&future.id) {
            Some(FutureState::Deferred(_))
                if !state.tasks.values().any(|task| task.future_id == future.id) => {}
            Some(_) => {
                return Err(ExecutionServiceError::Failed(
                    "future has already been scheduled".into(),
                ));
            }
            None => return Err(ExecutionServiceError::UnknownHandle),
        }
        let sequence = state.next_task;
        state.next_task = state.next_task.wrapping_add(1);
        let id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let generation = 1;
        state.tasks.insert(
            id,
            TaskRecord {
                future_id: future.id,
                generation,
                cancelled: false,
                read: false,
                claimed: false,
                completion_order: Some(sequence),
            },
        );
        Ok(TaskHandle {
            id,
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
        let state = self.state.lock().expect("execution service state poisoned");
        let stored_state = state
            .futures
            .get(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state: future_state(stored_state),
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
        let state = self.state.lock().expect("execution service state poisoned");
        let record = state
            .tasks
            .get(&task.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.generation != task.generation {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let handle_state = if record.cancelled {
            ExecutionHandleState::Cancelled
        } else {
            state
                .futures
                .get(&record.future_id)
                .map(future_state)
                .ok_or(ExecutionServiceError::UnknownHandle)?
        };
        Ok(ExecutionHandleSnapshot {
            state: handle_state,
            outputs: task.outputs.clone(),
            read: record.read,
        })
    }

    fn claim_next_task_result(
        &self,
        tasks: &[TaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        let mut state = self.state.lock().expect("execution service state poisoned");
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
        let mut state = self.state.lock().expect("execution service state poisoned");
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
        if let Value::Job(handle) = &value {
            return self.await_job(handle).map(AwaitAction::Completed);
        }
        let future = match value {
            Value::Future(handle) => handle,
            Value::Task(task) => {
                self.validate_scope(task.scope_id)?;
                let state = self.state.lock().expect("execution service state poisoned");
                let record = state
                    .tasks
                    .get(&task.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                if record.generation != task.generation {
                    return Err(ExecutionServiceError::UnknownHandle);
                }
                if record.cancelled {
                    return Err(ExecutionServiceError::Cancelled);
                }
                FutureHandle {
                    id: record.future_id,
                    scope_id: self.scope_id,
                    outputs: task.outputs,
                }
            }
            value => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        let record = state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        match record {
            FutureState::Deferred(call) => {
                let call = call.clone();
                *record = FutureState::Running;
                Ok(AwaitAction::ExecuteFuture {
                    handle: future,
                    call,
                })
            }
            FutureState::Running => Err(ExecutionServiceError::Failed(
                "execution is already being awaited".to_string(),
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
        let mut state = self.state.lock().expect("execution service state poisoned");
        let record = state
            .futures
            .get_mut(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if !matches!(record, FutureState::Running) {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        *record = FutureState::Completed(result);
        record_completion(&mut state, future.id);
        Ok(())
    }

    fn cancel(
        &self,
        value: &Value,
        _reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        let mut state = self.state.lock().expect("execution service state poisoned");
        match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                let record = state
                    .futures
                    .get_mut(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?;
                *record = FutureState::Cancelled;
                record_completion(&mut state, handle.id);
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let future_id = {
                    let record = state
                        .tasks
                        .get_mut(&handle.id)
                        .ok_or(ExecutionServiceError::UnknownHandle)?;
                    if record.generation != handle.generation {
                        return Err(ExecutionServiceError::UnknownHandle);
                    }
                    record.cancelled = true;
                    record.future_id
                };
                if let Some(future) = state.futures.get_mut(&future_id) {
                    *future = FutureState::Cancelled;
                }
                record_completion(&mut state, future_id);
            }
            _ => return Err(ExecutionServiceError::UnknownHandle),
        }
        Ok(())
    }

    fn drain_scope(&self, _reason: CancellationReason) {
        let mut state = self.state.lock().expect("execution service state poisoned");
        let mut completed = Vec::new();
        for future in state.futures.values_mut() {
            if !matches!(future, FutureState::Completed(_)) {
                *future = FutureState::Cancelled;
            }
        }
        for task in state.tasks.values_mut() {
            task.cancelled = true;
            completed.push(task.future_id);
        }
        for future_id in completed {
            record_completion(&mut state, future_id);
        }
    }
}

fn record_completion(state: &mut ServiceState, future_id: FutureId) {
    let Some(task_id) = state
        .tasks
        .iter()
        .find_map(|(task_id, task)| (task.future_id == future_id).then_some(*task_id))
    else {
        return;
    };
    if state
        .tasks
        .get(&task_id)
        .is_some_and(|task| task.completion_order.is_some())
    {
        return;
    }
    let order = state.next_completion;
    state.next_completion = order.wrapping_add(1);
    state
        .tasks
        .get_mut(&task_id)
        .expect("task was resolved above")
        .completion_order = Some(order);
}

fn future_state(state: &FutureState) -> ExecutionHandleState {
    match state {
        FutureState::Deferred(_) => ExecutionHandleState::Deferred,
        FutureState::Running => ExecutionHandleState::Running,
        FutureState::Completed(Ok(_)) => ExecutionHandleState::Finished,
        FutureState::Completed(Err(_)) => ExecutionHandleState::Failed,
        FutureState::Cancelled => ExecutionHandleState::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deferred_call(function: usize, arguments: Vec<Value>) -> DeferredCall {
        DeferredCall {
            invocation: DeferredInvocation::Callable(
                crate::call::descriptor::CallableDescriptor::resolved(
                    runmat_types::CallableIdentity::BoundFunction(runmat_types::FunctionId(
                        function,
                    )),
                    arguments,
                    1,
                    runmat_types::CallableFallbackPolicy::None,
                    crate::call::descriptor::CallableCallKind::Direct,
                ),
            ),
            retry: runmat_execution::RetryPolicy::Never,
            program_revision: None,
            program: None,
        }
    }

    #[test]
    fn services_reject_handles_from_another_scope() {
        let first = RuntimeExecutionService::new();
        let second = RuntimeExecutionService::new();
        let future = first.create_future(deferred_call(1, vec![])).unwrap();
        assert_eq!(
            second.spawn(&future),
            Err(ExecutionServiceError::ForeignScope)
        );
    }

    #[test]
    fn future_task_lifecycle_is_backed_by_service_state() {
        let service = RuntimeExecutionService::new();
        let future = service
            .create_future(deferred_call(7, vec![Value::Num(3.0)]))
            .unwrap();
        let task = service.spawn(&future).unwrap();
        let action = service.begin_await(Value::Task(task.clone())).unwrap();
        let AwaitAction::ExecuteFuture { handle, call } = action else {
            panic!("expected deferred execution");
        };
        assert_eq!(handle, future);
        assert!(matches!(
            call.invocation,
            DeferredInvocation::Callable(crate::call::descriptor::CallableDescriptor {
                target: crate::call::descriptor::CallableTarget::Resolved {
                    identity: runmat_types::CallableIdentity::BoundFunction(
                        runmat_types::FunctionId(7)
                    ),
                    ..
                },
                ..
            })
        ));
        service
            .complete_future(&future, Ok(Value::Num(9.0)))
            .unwrap();
        assert_eq!(
            service.begin_await(Value::Task(task)),
            Ok(AwaitAction::Completed(Value::Num(9.0)))
        );
    }

    #[test]
    fn cloned_invocation_context_inherits_exact_service() {
        let service: std::rc::Rc<dyn RuntimeExecutionServices> =
            std::rc::Rc::new(RuntimeExecutionService::new());
        let parent = crate::context::RuntimeContext::new(service);
        let nested = parent.clone();
        assert_eq!(parent.execution().scope_id(), nested.execution().scope_id());
    }

    #[test]
    fn independently_created_thread_sessions_have_distinct_scopes() {
        let first = std::thread::spawn(|| RuntimeExecutionService::new().scope_id())
            .join()
            .unwrap();
        let second = std::thread::spawn(|| RuntimeExecutionService::new().scope_id())
            .join()
            .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn scope_drain_cancels_unfinished_children() {
        let service = RuntimeExecutionService::new();
        let future = service.create_future(deferred_call(1, vec![])).unwrap();
        service.drain_scope(CancellationReason::Shutdown);
        assert_eq!(
            service.begin_await(Value::Future(future)),
            Err(ExecutionServiceError::Cancelled)
        );
    }

    #[test]
    fn serial_pool_lifecycle_fences_stale_handles() {
        let service = RuntimeExecutionService::new();
        assert!(service.current_pool().unwrap().is_none());
        let first = service.ensure_pool(PoolRequest::automatic()).unwrap();
        assert_eq!(first.backend, PoolBackend::Serial);
        assert_eq!(first.workers, 1);
        service.close_pool(&first.handle).unwrap();
        assert!(service.current_pool().unwrap().is_none());
        assert_eq!(
            service.close_pool(&first.handle),
            Err(ExecutionServiceError::UnknownHandle)
        );
        let reopened = service.ensure_pool(PoolRequest::automatic()).unwrap();
        assert_ne!(first.handle.generation, reopened.handle.generation);
    }
}
