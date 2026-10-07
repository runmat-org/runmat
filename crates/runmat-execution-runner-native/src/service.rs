use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::{
    CancellationReason, ExecutionScopeId, FutureHandle, FutureId, JobHandle, OutputContract,
    TaskHandle, TaskId,
};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, DurableJobOptions, ExecutionServiceError, RuntimeExecutionServices,
};
use runmat_value::Value;

use crate::config::fresh_scope_id;
use crate::driver::{LocalDriver, TaskCompletion};
use crate::durable::DurableJobBridge;
use crate::{NativeExecutionConfig, NativeExecutionResult};

static NEXT_NATIVE_SCOPE: AtomicU64 = AtomicU64::new(1);

mod state;
use state::{FutureRecord, FutureState, State, TaskRecord};

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
    ) -> Result<Result<(FutureHandle, Option<TaskId>), Value>, ExecutionServiceError> {
        match value {
            Value::Future(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.lock().expect("native service poisoned");
                Ok(Ok((state.future(&handle)?.handle.clone(), None)))
            }
            Value::Task(handle) => {
                self.validate_scope(handle.scope_id)?;
                let state = self.state.lock().expect("native service poisoned");
                Ok(Ok((state.task_future(&handle)?.clone(), Some(handle.id))))
            }
            value => Ok(Err(value)),
        }
    }
}

impl RuntimeExecutionServices for NativeExecutionService {
    fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    fn requires_program_capture(&self) -> bool {
        true
    }

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let sequence = state.next_future;
        state.next_future = sequence
            .checked_add(1)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        let id = FutureId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let handle = FutureHandle {
            id,
            scope_id: self.scope_id,
            outputs: OutputContract { requested_outputs },
        };
        state.futures.insert(
            id,
            FutureRecord {
                handle: handle.clone(),
                state: FutureState::Deferred(call),
            },
        );
        Ok(handle)
    }

    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let record = state.future(future)?;
        let call = match &record.state {
            FutureState::Deferred(call) => call.clone(),
            _ => {
                return Err(ExecutionServiceError::Failed(
                    "future has already been scheduled".into(),
                ))
            }
        };
        let outputs = record.handle.outputs.clone();
        let program = materialize_call(&call, outputs.clone())?;
        let sequence = state.next_task;
        state.next_task = sequence
            .checked_add(1)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        let id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let completion = self
            .driver
            .submit(id, program)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        state.future_mut(future)?.state = FutureState::Running(completion);
        let handle = TaskHandle {
            id,
            scope_id: self.scope_id,
            generation: 1,
            outputs,
        };
        state.tasks.insert(
            id,
            TaskRecord {
                handle: handle.clone(),
                future_id: future.id,
            },
        );
        Ok(handle)
    }

    fn submit_job(
        &self,
        call: DeferredCall,
        options: DurableJobOptions,
    ) -> Result<JobHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let outputs = OutputContract { requested_outputs };
        self.durable
            .submit(materialize_call(&call, outputs)?, options)
    }

    fn await_job(&self, job: &JobHandle) -> Result<Value, ExecutionServiceError> {
        self.durable.await_job(job.clone())
    }

    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        if let Value::Job(handle) = &value {
            return self
                .durable
                .try_await_job(handle)
                .map(|result| match result {
                    Some(value) => AwaitAction::Completed(value),
                    None => AwaitAction::Pending(value),
                });
        }
        let original = value.clone();
        let (future, task_id) = match self.future_for_value(value)? {
            Ok(future) => future,
            Err(value) => return Ok(AwaitAction::Passthrough(value)),
        };
        self.validate_scope(future.scope_id)?;
        let completion = {
            let mut state = self.state.lock().expect("native service poisoned");
            let record = state.future_mut(&future)?;
            match &mut record.state {
                FutureState::Deferred(_) => {
                    let FutureState::Deferred(call) =
                        std::mem::replace(&mut record.state, FutureState::ExecutingInCaller)
                    else {
                        unreachable!("deferred state matched while holding its owner lock")
                    };
                    return Ok(AwaitAction::ExecuteFuture {
                        handle: future,
                        call: Box::new(call),
                    });
                }
                FutureState::ExecutingInCaller => {
                    return Err(ExecutionServiceError::Failed(
                        "future is already executing in its caller".into(),
                    ))
                }
                FutureState::Running(completion) => Arc::clone(completion),
                FutureState::Completed(result) => {
                    let result = result.clone();
                    if let Some(task_id) = task_id {
                        state.tasks.remove(&task_id);
                    }
                    return result.map(AwaitAction::Completed);
                }
                FutureState::Cancelled => return Err(ExecutionServiceError::Cancelled),
            }
        };
        let Some(outcome) = completion.try_value() else {
            return Ok(AwaitAction::Pending(original));
        };
        let result = outcome
            .and_then(|success| {
                if !success.result_objects.is_empty() {
                    return Err(
                        "native runtime call returned externalized objects without an artifact consumer"
                            .into(),
                    );
                }
                runmat_runtime::execution::value_codec::decode_inline_outputs(&success.outputs, usize::from(future.outputs.requested_outputs))
                    .map_err(|error| error.to_string())
            })
            .map_err(ExecutionServiceError::Failed);
        let result = if completion.try_execution_evidence().is_some_and(|evidence| {
            evidence.task.state == runmat_execution::state::TaskState::Cancelled
        }) && completion
            .try_checkpoint_evidence()
            .is_some_and(|checkpoint| checkpoint.delivery_error().is_none())
        {
            Err(ExecutionServiceError::Cancelled)
        } else {
            result
        };
        let mut state = self.state.lock().expect("native service poisoned");
        let record = state.future_mut(&future)?;
        let result = match &record.state {
            FutureState::Cancelled => return Err(ExecutionServiceError::Cancelled),
            FutureState::Completed(committed) => committed.clone(),
            FutureState::Running(original) if Arc::ptr_eq(original, &completion) => {
                record.state = FutureState::Completed(result.clone());
                result
            }
            _ => return Err(ExecutionServiceError::UnknownHandle),
        };
        if let Some(task_id) = task_id {
            state.tasks.remove(&task_id);
        }
        result.map(AwaitAction::Completed)
    }

    fn complete_future(
        &self,
        future: &FutureHandle,
        result: Result<Value, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.validate_scope(future.scope_id)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let record = state.future_mut(future)?;
        match record.state {
            FutureState::ExecutingInCaller => {}
            FutureState::Cancelled => return Err(ExecutionServiceError::Cancelled),
            _ => {
                return Err(ExecutionServiceError::Failed(
                    "future is not executing in its caller".into(),
                ))
            }
        }
        if let Ok(value) = &result {
            runmat_runtime::execution::value_codec::validate_output_arity(
                value,
                usize::from(record.handle.outputs.requested_outputs),
            )
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        }
        record.state = FutureState::Completed(result);
        Ok(())
    }

    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        if let Value::Job(handle) = value {
            return self.durable.cancel(handle.clone(), reason);
        }
        let (future, task_id) = self
            .future_for_value(value.clone())?
            .map_err(|_| ExecutionServiceError::UnknownHandle)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let record = state.future_mut(&future)?;
        if matches!(record.state, FutureState::Completed(_)) {
            return Ok(());
        }
        if let FutureState::Running(_) = &record.state {
            let id = task_id
                .or_else(|| {
                    state
                        .tasks
                        .iter()
                        .find_map(|(id, task)| (task.future_id == future.id).then_some(*id))
                })
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            self.driver
                .cancel_task(id, reason)
                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
            // The shared driver owns terminal outcome and quiescence. Keep polling
            // the original completion; a cancellation request is not a terminal report.
            return Ok(());
        }
        record.state = FutureState::Cancelled;
        Ok(())
    }

    fn drain_scope(&self, reason: CancellationReason) {
        self.driver.cancel_all(reason);
        let mut state = self.state.lock().expect("native service poisoned");
        for future in state.futures.values_mut() {
            if let FutureState::Running(completion) = &future.state {
                completion.cancel();
            }
            if !matches!(
                future.state,
                FutureState::Completed(_) | FutureState::Running(_)
            ) {
                future.state = FutureState::Cancelled;
            }
        }
    }
}

fn materialize_call(
    call: &DeferredCall,
    outputs: OutputContract,
) -> Result<runmat_execution_artifact::ProgramExecutionRequest, ExecutionServiceError> {
    let request = runmat_vm::materialize_deferred_call(
        call,
        outputs.clone(),
        runmat_execution_artifact::ProgramTarget::portable(format!(
            "{}-{}-interpreter-bytecode-v1",
            std::env::consts::ARCH,
            std::env::consts::OS,
        )),
    )?;
    request
        .validate_for_portable_host()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    Ok(request)
}
