use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionHandleState, ExecutionScopeId,
    FutureHandle, FutureId, JobHandle, OutputContract, PoolBackend, PoolHandle, PoolRequest,
    PoolSnapshot, PoolState, TaskHandle, TaskId, TaskResultClaim,
};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, DurableJobOptions, ExecutionServiceError, RuntimeExecutionServices,
};
use runmat_value::Value;

use crate::config::fresh_scope_id;
use crate::driver::{LocalDriver, TaskCompletion};
use crate::durable::DurableJobBridge;
use crate::supervisor::ProgramBatchSubmission;
use crate::{NativeExecutionConfig, NativeExecutionResult};

static NEXT_NATIVE_SCOPE: AtomicU64 = AtomicU64::new(1);

enum FutureState {
    Deferred(Box<DeferredCall>),
    Running(Arc<TaskCompletion>),
    Completed(Result<Value, ExecutionServiceError>),
    Cancelled,
}

struct TaskRecord {
    future_id: FutureId,
    generation: u64,
    read: bool,
    claimed: bool,
    completion_order: Option<u64>,
}

struct State {
    next_future: u64,
    next_task: u64,
    futures: HashMap<FutureId, FutureState>,
    tasks: HashMap<TaskId, TaskRecord>,
    pool_generation: u64,
    pool_open: bool,
}

pub struct NativeExecutionService {
    scope_id: ExecutionScopeId,
    driver: Arc<LocalDriver>,
    durable: DurableJobBridge,
    state: Mutex<State>,
}

impl NativeExecutionService {
    pub fn new(mut config: NativeExecutionConfig) -> NativeExecutionResult<Self> {
        let nonce = NEXT_NATIVE_SCOPE.fetch_add(1, Ordering::Relaxed);
        let scope_id = fresh_scope_id(b"native-session", nonce);
        config.store_root.push(scope_id.to_string());
        Ok(Self {
            scope_id,
            driver: LocalDriver::new(config, scope_id)?,
            durable: DurableJobBridge::start()
                .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?,
            state: Mutex::new(State {
                next_future: 0,
                next_task: 0,
                futures: HashMap::new(),
                tasks: HashMap::new(),
                pool_generation: 1,
                pool_open: false,
            }),
        })
    }

    fn validate_scope(&self, scope_id: ExecutionScopeId) -> Result<(), ExecutionServiceError> {
        if scope_id == self.scope_id {
            Ok(())
        } else {
            Err(ExecutionServiceError::ForeignScope)
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
                let state = self.state.lock().expect("native service poisoned");
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
}

impl RuntimeExecutionServices for NativeExecutionService {
    fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        let state = self.state.lock().expect("native service poisoned");
        Ok(state.pool_open.then(|| PoolSnapshot {
            handle: PoolHandle {
                id: self.driver.pool_id(),
                scope_id: self.scope_id,
                generation: state.pool_generation,
            },
            backend: PoolBackend::LocalProcesses,
            workers: self.driver.max_workers(),
            state: PoolState::Ready,
        }))
    }

    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        if request
            .backend
            .is_some_and(|backend| backend != PoolBackend::LocalProcesses)
            || request
                .workers
                .is_some_and(|workers| workers != self.driver.max_workers())
        {
            return Err(ExecutionServiceError::Failed(format!(
                "this session owns a fixed local process pool with {} workers",
                self.driver.max_workers()
            )));
        }
        let mut state = self.state.lock().expect("native service poisoned");
        state.pool_open = true;
        Ok(PoolSnapshot {
            handle: PoolHandle {
                id: self.driver.pool_id(),
                scope_id: self.scope_id,
                generation: state.pool_generation,
            },
            backend: PoolBackend::LocalProcesses,
            workers: self.driver.max_workers(),
            state: PoolState::Ready,
        })
    }

    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.validate_scope(pool.scope_id)?;
        let mut state = self.state.lock().expect("native service poisoned");
        if !state.pool_open
            || pool.id != self.driver.pool_id()
            || pool.generation != state.pool_generation
        {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        state.pool_open = false;
        state.pool_generation = state.pool_generation.wrapping_add(1);
        drop(state);
        self.driver.cancel_all(CancellationReason::User);
        Ok(())
    }

    fn requires_program_capture(&self) -> bool {
        true
    }

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.descriptor.requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("native service poisoned");
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
        let mut state = self.state.lock().expect("native service poisoned");
        let call = match state.futures.get(&future.id) {
            Some(FutureState::Deferred(call)) => call.clone(),
            Some(_) => {
                return Err(ExecutionServiceError::Failed(
                    "future has already been scheduled".into(),
                ))
            }
            None => return Err(ExecutionServiceError::UnknownHandle),
        };
        let (callable, recipe, artifact, inputs) = materialize_call(&call, future.outputs.clone())?;
        let sequence = state.next_task;
        state.next_task = sequence.wrapping_add(1);
        let id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let completion = self
            .driver
            .submit(
                id,
                callable,
                recipe,
                artifact,
                inputs,
                future.outputs.clone(),
            )
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        state
            .futures
            .insert(future.id, FutureState::Running(completion));
        let generation = 1;
        state.tasks.insert(
            id,
            TaskRecord {
                future_id: future.id,
                generation,
                read: false,
                claimed: false,
                completion_order: None,
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
        let state = self.state.lock().expect("native service poisoned");
        let stored_state = state
            .futures
            .get(&future.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        Ok(ExecutionHandleSnapshot {
            state: native_future_state(stored_state),
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
        let state = self.state.lock().expect("native service poisoned");
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
            .map(native_future_state)
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
        let mut state = self.state.lock().expect("native service poisoned");
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
            let order =
                record
                    .completion_order
                    .or_else(|| match state.futures.get(&record.future_id) {
                        Some(FutureState::Running(completion)) => completion.completion_order(),
                        _ => None,
                    });
            if let Some(order) = order {
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
        let mut state = self.state.lock().expect("native service poisoned");
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

    fn submit_job(
        &self,
        call: DeferredCall,
        options: DurableJobOptions,
    ) -> Result<JobHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.descriptor.requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let outputs = OutputContract { requested_outputs };
        let (callable, recipe, artifact, arguments) = materialize_call(&call, outputs)?;
        self.durable.submit(ProgramBatchSubmission {
            recipe,
            artifact,
            callable,
            arguments,
            requested_outputs,
            idempotency_key: options.idempotency_key,
            retention_millis: options.retention_millis,
        })
    }

    fn await_job(&self, job: &JobHandle) -> Result<Value, ExecutionServiceError> {
        self.durable.await_job(job.clone())
    }

    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        if let Value::Job(handle) = &value {
            return self.await_job(handle).map(AwaitAction::Completed);
        }
        let original = value.clone();
        let future = match self.future_for_value(value)? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let completion = {
            let mut state = self.state.lock().expect("native service poisoned");
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
                        .expect("future was just resolved") = FutureState::Completed(Err(
                        ExecutionServiceError::Failed("future is executing in its caller".into()),
                    ));
                    return Ok(AwaitAction::ExecuteFuture {
                        handle: future,
                        call,
                    });
                }
                FutureState::Running(completion) => Arc::clone(completion),
                FutureState::Completed(result) => {
                    let result = result.clone();
                    return result.map(AwaitAction::Completed);
                }
                FutureState::Cancelled => return Err(ExecutionServiceError::Cancelled),
            }
        };
        let Some(completion) = completion.try_value() else {
            return Ok(AwaitAction::Pending(original));
        };
        let result = completion
            .and_then(|success| {
                let [payload] = success.outputs.as_slice() else {
                    return Err(
                        "native runtime call did not return exactly one output value".into(),
                    );
                };
                if !success.result_objects.is_empty() {
                    return Err(
                        "native runtime call returned externalized objects without an artifact consumer"
                            .into(),
                    );
                }
                runmat_runtime::execution::value_codec::decode_inline_value(payload)
                    .map_err(|error| error.to_string())
            })
            .map_err(ExecutionServiceError::Failed);
        let mut state = self.state.lock().expect("native service poisoned");
        state
            .futures
            .insert(future.id, FutureState::Completed(result.clone()));
        result.map(AwaitAction::Completed)
    }

    fn complete_future(
        &self,
        future: &FutureHandle,
        result: Result<Value, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        self.state
            .lock()
            .expect("native service poisoned")
            .futures
            .insert(future.id, FutureState::Completed(result));
        let mut state = self.state.lock().expect("native service poisoned");
        record_native_completion(&mut state, future.id);
        Ok(())
    }

    fn cancel(
        &self,
        value: &Value,
        _reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        let future_id = match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                handle.id
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                self.state
                    .lock()
                    .expect("native service poisoned")
                    .tasks
                    .get(&handle.id)
                    .ok_or(ExecutionServiceError::UnknownHandle)?
                    .future_id
            }
            Value::Job(handle) => return self.durable.cancel(handle.clone()),
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        let mut state = self.state.lock().expect("native service poisoned");
        let record = state
            .futures
            .get_mut(&future_id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if let FutureState::Running(completion) = record {
            completion.cancel();
        }
        *record = FutureState::Cancelled;
        record_native_completion(&mut state, future_id);
        Ok(())
    }

    fn drain_scope(&self, reason: CancellationReason) {
        self.driver.cancel_all(reason);
        let mut state = self.state.lock().expect("native service poisoned");
        for future in state.futures.values_mut() {
            if let FutureState::Running(completion) = future {
                completion.cancel();
            }
            if !matches!(future, FutureState::Completed(_)) {
                *future = FutureState::Cancelled;
            }
        }
        let future_ids = state
            .tasks
            .values()
            .map(|task| task.future_id)
            .collect::<Vec<_>>();
        for future_id in future_ids {
            record_native_completion(&mut state, future_id);
        }
    }
}

fn record_native_completion(state: &mut State, future_id: FutureId) {
    let Some(task) = state
        .tasks
        .values_mut()
        .find(|task| task.future_id == future_id)
    else {
        return;
    };
    if task.completion_order.is_none() {
        task.completion_order = Some(crate::driver::next_task_completion_order());
    }
}

fn native_future_state(state: &FutureState) -> ExecutionHandleState {
    match state {
        FutureState::Deferred(_) => ExecutionHandleState::Deferred,
        FutureState::Running(completion) if completion.try_value().is_none() => {
            ExecutionHandleState::Running
        }
        FutureState::Running(completion) => match completion.try_value() {
            Some(Ok(_)) => ExecutionHandleState::Finished,
            Some(Err(_)) => ExecutionHandleState::Failed,
            None => ExecutionHandleState::Running,
        },
        FutureState::Completed(Ok(_)) => ExecutionHandleState::Finished,
        FutureState::Completed(Err(_)) => ExecutionHandleState::Failed,
        FutureState::Cancelled => ExecutionHandleState::Cancelled,
    }
}

fn materialize_call(
    call: &DeferredCall,
    outputs: OutputContract,
) -> Result<
    (
        runmat_execution::ProgramCallable,
        runmat_execution_artifact::ProgramBuildRecipe,
        runmat_execution_artifact::ProgramArtifact,
        Vec<runmat_execution::value::ValuePayload>,
    ),
    ExecutionServiceError,
> {
    runmat_vm::materialize_deferred_call(
        call,
        outputs,
        runmat_execution_artifact::ProgramTarget::portable(format!(
            "{}-{}-interpreter-bytecode-v1",
            std::env::consts::ARCH,
            std::env::consts::OS
        )),
    )
}
