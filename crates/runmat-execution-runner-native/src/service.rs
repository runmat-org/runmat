use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_builtins::Value;
use runmat_execution::{
    CancellationReason, Digest, ExecutionScopeId, FutureHandle, FutureId, OutputContract,
    ProgramEnvironment, ProgramRevision, TaskHandle, TaskId,
};
use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, ExecutionServiceError, RuntimeExecutionServices,
};

use crate::driver::{LocalDriver, TaskCompletion};
use crate::{NativeExecutionConfig, NativeExecutionResult};

static NEXT_NATIVE_SCOPE: AtomicU64 = AtomicU64::new(1);

enum FutureState {
    Deferred(DeferredCall),
    Running(Arc<TaskCompletion>),
    Completed(Result<Value, ExecutionServiceError>),
    Cancelled,
}

struct TaskRecord {
    future_id: FutureId,
    generation: u64,
}

struct State {
    next_future: u64,
    next_task: u64,
    futures: HashMap<FutureId, FutureState>,
    tasks: HashMap<TaskId, TaskRecord>,
}

pub struct NativeExecutionService {
    scope_id: ExecutionScopeId,
    driver: Arc<LocalDriver>,
    state: Mutex<State>,
}

impl NativeExecutionService {
    pub fn new(mut config: NativeExecutionConfig) -> NativeExecutionResult<Self> {
        let nonce = NEXT_NATIVE_SCOPE.fetch_add(1, Ordering::Relaxed);
        let scope_id = ExecutionScopeId::derive(&[
            b"native-session",
            &std::process::id().to_be_bytes(),
            &nonce.to_be_bytes(),
        ]);
        config.store_root.push(scope_id.to_string());
        Ok(Self {
            scope_id,
            driver: LocalDriver::new(config, scope_id)?,
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

    fn requires_program_capture(&self) -> bool {
        true
    }

    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let requested_outputs = u16::try_from(call.requested_outputs)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        let mut state = self.state.lock().expect("native service poisoned");
        let sequence = state.next_future;
        state.next_future = sequence.wrapping_add(1);
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
        let program = call.program.as_deref().ok_or_else(|| {
            ExecutionServiceError::Failed("future is missing its exact program".into())
        })?;
        let revision = call
            .program_revision
            .clone()
            .unwrap_or_else(|| local_program_revision(program));
        let recipe = ProgramBuildRecipe {
            schema_version: 1,
            program_revision: revision,
            entrypoint: call.function.to_string(),
            outputs: future.outputs.clone(),
            execution_mode: "interpreter".into(),
            target_profile: format!(
                "{}-{}-interpreter-bytecode-v1",
                std::env::consts::ARCH,
                std::env::consts::OS
            ),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV1,
            program.to_vec(),
        )
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        let inputs = call
            .arguments
            .iter()
            .map(runmat_runtime::execution::value_codec::encode_inline_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        let sequence = state.next_task;
        state.next_task = sequence.wrapping_add(1);
        let id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let completion = self
            .driver
            .submit(
                id,
                call.function,
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
                    return result.clone().map(AwaitAction::Completed)
                }
                FutureState::Cancelled => return Err(ExecutionServiceError::Cancelled),
            }
        };
        let result = completion
            .wait()
            .and_then(|payload| {
                runmat_runtime::execution::value_codec::decode_inline_value(&payload)
                    .map_err(|error| error.to_string())
            })
            .map_err(ExecutionServiceError::Failed);
        self.state
            .lock()
            .expect("native service poisoned")
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
    }
}

fn local_program_revision(program: &[u8]) -> ProgramRevision {
    let digest = Digest::sha256(program);
    ProgramRevision::new(
        digest,
        digest,
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(format!(
                "runmat-runtime-abi-v1\0{}",
                env!("CARGO_PKG_VERSION")
            )),
            Digest::sha256(b"runmat-local-captured-catalog-v1"),
            "matlab",
        )
        .expect("native execution compatibility constants are valid"),
    )
    .expect("captured local program revision is valid")
}
