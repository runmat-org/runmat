use runmat_native_codegen::NativeEdge;
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, ExecutionServiceError, RootedValue, RootedValueSequence,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;
use runmat_runtime::execution::RunningFutureLease;

pub(super) enum AwaitStart {
    Ready(RootedValue),
    Suspended { continuation: u64, generation: u64 },
}

pub(super) struct AwaitCompletion {
    pub edge: NativeEdge,
    pub value: RootedValue,
}

pub(super) struct PendingAwait {
    continuation: u64,
    generation: u64,
    edge: NativeEdge,
    work: AwaitWork,
}

enum AwaitWork {
    Poll(Value),
    Execute {
        running: Box<RunningFutureLease>,
        call: Box<DeferredCall>,
    },
}

pub(super) fn begin(
    state: &mut HostState,
    value: Value,
    edge: NativeEdge,
) -> NativeExecutorResult<AwaitStart> {
    let action = state
        .runtime
        .execution()
        .begin_await_in_context(value, &state.runtime)
        .map_err(execution_error)?;
    match action {
        AwaitAction::Passthrough(value) => Ok(AwaitStart::Ready(root_value(value)?)),
        AwaitAction::Completed(sequence) => Ok(AwaitStart::Ready(resolve_await_value(sequence)?)),
        AwaitAction::Pending(value) => {
            let (continuation, generation) = state.next_suspension_identity()?;
            state.pending_await = Some(PendingAwait {
                continuation,
                generation,
                edge,
                work: AwaitWork::Poll(value),
            });
            Ok(AwaitStart::Suspended {
                continuation,
                generation,
            })
        }
        AwaitAction::ExecuteFuture { handle, call } => {
            let running = Box::new(RunningFutureLease::new(state.runtime.clone(), handle));
            let (continuation, generation) = state.next_suspension_identity()?;
            state.pending_await = Some(PendingAwait {
                continuation,
                generation,
                edge,
                work: AwaitWork::Execute { running, call },
            });
            Ok(AwaitStart::Suspended {
                continuation,
                generation,
            })
        }
    }
}

pub(super) async fn complete(
    state: &mut HostState,
    continuation: u64,
    generation: u64,
) -> NativeExecutorResult<AwaitCompletion> {
    validate_pending(state, continuation, generation)?;
    let pending = state.pending_await.take().ok_or_else(|| {
        NativeExecutorError::Host("native invocation has no pending await".into())
    })?;
    let runtime = state.runtime.clone();
    let edge = pending.edge;
    let mut work = pending.work;
    loop {
        match work {
            AwaitWork::Poll(value) => {
                if let Some(waiting) = runtime
                    .execution()
                    .wait_pending(&value, &runtime)
                    .map_err(execution_error)?
                {
                    waiting.await.map_err(execution_error)?;
                } else {
                    yield_once().await;
                }
                match runtime
                    .execution()
                    .begin_await_in_context(value, &runtime)
                    .map_err(execution_error)?
                {
                    AwaitAction::Passthrough(value) => {
                        return Ok(AwaitCompletion {
                            edge,
                            value: root_value(value)?,
                        });
                    }
                    AwaitAction::Completed(sequence) => {
                        return Ok(AwaitCompletion {
                            edge,
                            value: resolve_await_value(sequence)?,
                        });
                    }
                    AwaitAction::Pending(value) => work = AwaitWork::Poll(value),
                    AwaitAction::ExecuteFuture { handle, call } => {
                        work = AwaitWork::Execute {
                            running: Box::new(RunningFutureLease::new(runtime.clone(), handle)),
                            call,
                        };
                    }
                }
            }
            AwaitWork::Execute { mut running, call } => {
                let requested_outputs = call.invocation.requested_outputs();
                let result = execute_deferred_call(&runtime, *call)
                    .await
                    .and_then(|sequence| normalize_outputs(sequence, requested_outputs));
                let sequence = store_callable_outcome(&mut running, result)?;
                return Ok(AwaitCompletion {
                    edge,
                    value: resolve_await_value(sequence)?,
                });
            }
        }
    }
}

/// Run the deferred callable inside the original invocation environment it
/// captured at future creation, not the awaiting caller's current one.
async fn execute_deferred_call(
    runtime: &runmat_runtime::context::RuntimeContext,
    call: DeferredCall,
) -> NativeExecutorResult<RootedValueSequence> {
    let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) = call.invocation
    else {
        return Err(runmat_runtime::runtime_error::semantic_error(
            "ParallelProgramUnavailable",
            "native continuation cannot execute a deferred program in-process",
        )
        .into());
    };
    let invocation = runtime
        .clone()
        .with_invocation_environment(&call.environment)
        .map_err(|error| execution_error(ExecutionServiceError::Failed(error.to_string())))?;
    runmat_runtime::call::descriptor::execute_callable_descriptor_in_context(
        &invocation,
        *descriptor,
    )
    .await
    .map_err(NativeExecutorError::from)
}

fn store_callable_outcome(
    running: &mut RunningFutureLease,
    result: NativeExecutorResult<RootedValueSequence>,
) -> NativeExecutorResult<RootedValueSequence> {
    match result {
        Ok(sequence) => {
            running
                .complete(Ok(sequence.clone()))
                .map_err(execution_error)?;
            Ok(sequence)
        }
        Err(NativeExecutorError::Runtime(error)) => {
            let original = runmat_runtime::runtime_error::RuntimeErrorOwner::new(*error);
            running
                .complete(Err(ExecutionServiceError::LocalizedRuntimeFailure(
                    original.clone(),
                )))
                .map_err(execution_error)?;
            Err(runmat_runtime::RuntimeError::from_shared_original(original).into())
        }
        Err(error) => {
            running
                .complete(Err(ExecutionServiceError::Failed(error.to_string())))
                .map_err(execution_error)?;
            Err(error)
        }
    }
}

pub(super) fn validate_pending(
    state: &HostState,
    continuation: u64,
    generation: u64,
) -> NativeExecutorResult<()> {
    let pending = state.pending_await.as_ref().ok_or_else(|| {
        NativeExecutorError::Host("native invocation has no pending await".into())
    })?;
    if pending.continuation != continuation || pending.generation != generation {
        return Err(NativeExecutorError::Host(
            "native await continuation identity is stale or mismatched".into(),
        ));
    }
    Ok(())
}

fn normalize_outputs(
    sequence: RootedValueSequence,
    requested_outputs: usize,
) -> NativeExecutorResult<RootedValueSequence> {
    runmat_value::validate_output_count(requested_outputs)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
    if requested_outputs == 0 {
        return RootedValueSequence::new(runmat_value::ValueSequence::empty())
            .map_err(execution_error);
    }
    if sequence.len() != requested_outputs {
        return Err(runmat_runtime::runtime_error::semantic_error(
            "OutputArityMismatch",
            format!(
                "deferred call returned {} outputs for {requested_outputs} requested outputs",
                sequence.len()
            ),
        )
        .into());
    }
    Ok(sequence)
}

fn resolve_await_value(sequence: RootedValueSequence) -> Result<RootedValue, NativeExecutorError> {
    sequence
        .try_into_single()
        .map_err(NativeExecutorError::from)
}

fn root_value(value: Value) -> NativeExecutorResult<RootedValue> {
    RootedValue::new(value).map_err(execution_error)
}

fn execution_error(error: ExecutionServiceError) -> NativeExecutorError {
    if matches!(&error, ExecutionServiceError::LocalizedRuntimeFailure(_)) {
        return error.into_runtime_error().into();
    }
    runmat_runtime::runtime_error::semantic_error("ExecutionService", error.to_string()).into()
}

async fn yield_once() {
    let mut yielded = false;
    futures::future::poll_fn(|context| {
        if yielded {
            std::task::Poll::Ready(())
        } else {
            yielded = true;
            context.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    })
    .await;
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::rc::Rc;

    use runmat_runtime::context::RuntimeContext;
    use runmat_runtime::execution::{
        AwaitAction, DeferredInvocation, DeferredProgram, RuntimeExecutionService,
        RuntimeExecutionServices,
    };
    use runmat_types::{CallableFallbackPolicy, CallableIdentity, FunctionId};

    use super::*;

    #[test]
    fn local_callable_failure_keeps_one_cached_typed_error_owner() {
        let service = Rc::new(RuntimeExecutionService::new());
        let runtime = RuntimeContext::new(service.clone());
        let future = service
            .create_future(DeferredCall {
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
                invocation: DeferredInvocation::callable(
                    runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                        CallableIdentity::BoundFunction(FunctionId(1)),
                        Vec::new(),
                        1,
                        CallableFallbackPolicy::None,
                        runmat_runtime::call::descriptor::CallableCallKind::Direct,
                    ),
                ),
                retry: runmat_execution::RetryPolicy::Never,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                console: None,
                program_revision: None,
                capabilities: Default::default(),
                program: DeferredProgram::Local,
            })
            .expect("create local future");
        assert!(matches!(
            service.begin_await(Value::Future(future.clone().into())),
            Ok(AwaitAction::ExecuteFuture { .. })
        ));

        let error = runmat_runtime::build_runtime_error("local callback failed")
            .with_identifier("Acme:LocalCallback")
            .with_call_frames(vec![runmat_runtime::CallFrame {
                function: "local_callback".into(),
                source_id: None,
                span: Some((4, 2)),
                source_owner: None,
                diagnostic_source: None,
            }])
            .with_source(std::io::Error::other("typed provider failure"))
            .build();
        let mut running = RunningFutureLease::new(runtime, future.clone());
        let immediate = store_callable_outcome(&mut running, Err(error.into()))
            .expect_err("local callback must fail");
        let NativeExecutorError::Runtime(immediate) = immediate else {
            panic!("local failure must retain RuntimeError");
        };
        assert_eq!(immediate.identifier(), Some("Acme:LocalCallback"));
        assert_eq!(immediate.context.call_frames[0].function, "local_callback");
        assert!(has_typed_provider_source(immediate.as_ref()));

        let first = service
            .begin_await(Value::Future(future.clone().into()))
            .expect_err("first cached fetch must fail");
        let second = service
            .begin_await(Value::Future(future.into()))
            .expect_err("repeated cached fetch must fail");
        assert_eq!(first, second, "both fetches must share one failure owner");
        for failure in [first, second] {
            let error = failure.into_runtime_error();
            assert_eq!(error.identifier(), Some("Acme:LocalCallback"));
            assert_eq!(error.context.call_frames[0].function, "local_callback");
            assert!(has_typed_provider_source(&error));
        }
    }

    fn has_typed_provider_source(error: &(dyn Error + 'static)) -> bool {
        let mut source = error.source();
        while let Some(current) = source {
            if current.downcast_ref::<std::io::Error>().is_some() {
                return true;
            }
            source = current.source();
        }
        false
    }
}
