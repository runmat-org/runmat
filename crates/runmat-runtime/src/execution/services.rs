use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use runmat_execution::{
    CancellationReason, ExecutionScopeId, FutureHandle, FutureId, JobHandle, OutputContract,
    TaskHandle, TaskId,
};
use runmat_value::Value;

use super::ExecutionServiceError;

static NEXT_SERVICE_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq)]
pub struct DeferredCall {
    pub function: usize,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
    pub program_revision: Option<runmat_execution::ProgramRevision>,
    /// Exact, runtime-opaque program description supplied by the VM.
    ///
    /// The serial service does not inspect this payload. Execution adapters may
    /// require it to reproduce the callable in an isolated worker.
    pub program: Option<Vec<u8>>,
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
        call: DeferredCall,
    },
    Completed(Value),
}

pub trait RuntimeExecutionServices {
    fn scope_id(&self) -> ExecutionScopeId;
    fn requires_program_capture(&self) -> bool {
        false
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError>;
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError>;
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
    Deferred(DeferredCall),
    Running,
    Completed(Result<Value, ExecutionServiceError>),
    Cancelled,
}

#[derive(Clone, Debug)]
struct TaskRecord {
    future_id: FutureId,
    generation: u64,
    cancelled: bool,
}

#[derive(Debug)]
struct ServiceState {
    next_future: u64,
    next_task: u64,
    futures: HashMap<FutureId, FutureState>,
    tasks: HashMap<TaskId, TaskRecord>,
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
                futures: HashMap::new(),
                tasks: HashMap::new(),
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

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        let sequence = state.next_future;
        state.next_future = state.next_future.wrapping_add(1);
        let id = FutureId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        state.futures.insert(id, FutureState::Deferred(call));
        Ok(FutureHandle {
            id,
            scope_id: self.scope_id,
            outputs: OutputContract { requested_outputs },
        })
    }

    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("execution service state poisoned");
        if !state.futures.contains_key(&future.id) {
            return Err(ExecutionServiceError::UnknownHandle);
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
            },
        );
        Ok(TaskHandle {
            id,
            scope_id: self.scope_id,
            generation,
            outputs: future.outputs.clone(),
        })
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
            FutureState::Completed(result) => result.clone().map(AwaitAction::Completed),
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
            }
            _ => return Err(ExecutionServiceError::UnknownHandle),
        }
        Ok(())
    }

    fn drain_scope(&self, _reason: CancellationReason) {
        let mut state = self.state.lock().expect("execution service state poisoned");
        for future in state.futures.values_mut() {
            if !matches!(future, FutureState::Completed(_)) {
                *future = FutureState::Cancelled;
            }
        }
        for task in state.tasks.values_mut() {
            task.cancelled = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn services_reject_handles_from_another_scope() {
        let first = RuntimeExecutionService::new();
        let second = RuntimeExecutionService::new();
        let future = first
            .create_future(DeferredCall {
                function: 1,
                arguments: vec![],
                requested_outputs: 1,
                program_revision: None,
                program: None,
            })
            .unwrap();
        assert_eq!(
            second.spawn(&future),
            Err(ExecutionServiceError::ForeignScope)
        );
    }

    #[test]
    fn future_task_lifecycle_is_backed_by_service_state() {
        let service = RuntimeExecutionService::new();
        let future = service
            .create_future(DeferredCall {
                function: 7,
                arguments: vec![Value::Num(3.0)],
                requested_outputs: 1,
                program_revision: None,
                program: None,
            })
            .unwrap();
        let task = service.spawn(&future).unwrap();
        let action = service.begin_await(Value::Task(task.clone())).unwrap();
        let AwaitAction::ExecuteFuture { handle, call } = action else {
            panic!("expected deferred execution");
        };
        assert_eq!(handle, future);
        assert_eq!(call.function, 7);
        service
            .complete_future(&future, Ok(Value::Num(9.0)))
            .unwrap();
        assert_eq!(
            service.begin_await(Value::Task(task)).unwrap(),
            AwaitAction::Completed(Value::Num(9.0))
        );
    }

    #[test]
    fn cloned_invocation_context_inherits_exact_service() {
        let service: std::rc::Rc<dyn RuntimeExecutionServices> =
            std::rc::Rc::new(RuntimeExecutionService::new());
        let parent = crate::execution::InvocationExecutionContext::new(service);
        let nested = parent.clone();
        assert_eq!(parent.services().scope_id(), nested.services().scope_id());
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
        let future = service
            .create_future(DeferredCall {
                function: 1,
                arguments: vec![],
                requested_outputs: 1,
                program_revision: None,
                program: None,
            })
            .unwrap();
        service.drain_scope(CancellationReason::Shutdown);
        assert_eq!(
            service.begin_await(Value::Future(future)),
            Err(ExecutionServiceError::Cancelled)
        );
    }
}
