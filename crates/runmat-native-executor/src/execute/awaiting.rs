use runmat_native_codegen::NativeEdge;
use runmat_runtime::execution::{AwaitAction, DeferredCall, ExecutionServiceError};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

pub(super) enum AwaitStart {
    Ready(Value),
    Suspended { continuation: u64, generation: u64 },
}

pub(super) struct AwaitCompletion {
    pub edge: NativeEdge,
    pub value: Value,
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
        handle: runmat_execution::FutureHandle,
        call: DeferredCall,
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
        .begin_await(value)
        .map_err(execution_error)?;
    match action {
        AwaitAction::Passthrough(value) | AwaitAction::Completed(value) => {
            Ok(AwaitStart::Ready(value))
        }
        AwaitAction::Pending(value) => {
            let (continuation, generation) = state.next_await_identity()?;
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
            let (continuation, generation) = state.next_await_identity()?;
            state.pending_await = Some(PendingAwait {
                continuation,
                generation,
                edge,
                work: AwaitWork::Execute { handle, call },
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
    let pending = state.pending_await.take().ok_or_else(|| {
        NativeExecutorError::Host("native invocation has no pending await".into())
    })?;
    if pending.continuation != continuation || pending.generation != generation {
        state.pending_await = Some(pending);
        return Err(NativeExecutorError::Host(
            "native await continuation identity is stale or mismatched".into(),
        ));
    }
    let runtime = state.runtime.clone();
    let edge = pending.edge;
    let mut work = pending.work;
    loop {
        match work {
            AwaitWork::Poll(value) => {
                yield_once().await;
                match runtime
                    .execution()
                    .begin_await(value)
                    .map_err(execution_error)?
                {
                    AwaitAction::Passthrough(value) | AwaitAction::Completed(value) => {
                        return Ok(AwaitCompletion { edge, value });
                    }
                    AwaitAction::Pending(value) => work = AwaitWork::Poll(value),
                    AwaitAction::ExecuteFuture { handle, call } => {
                        work = AwaitWork::Execute { handle, call };
                    }
                }
            }
            AwaitWork::Execute { handle, call } => {
                let descriptor = runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                    runmat_hir::CallableIdentity::BoundFunction(runmat_hir::FunctionId(
                        call.function,
                    )),
                    call.arguments,
                    call.requested_outputs,
                    runmat_hir::CallableFallbackPolicy::None,
                    runmat_runtime::call::descriptor::CallableCallKind::Direct,
                );
                let result = runtime
                    .scope(
                        runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor),
                    )
                    .await
                    .map(|value| normalize_outputs(value, call.requested_outputs));
                let stored = result
                    .as_ref()
                    .map(Clone::clone)
                    .map_err(|error| ExecutionServiceError::Failed(error.to_string()));
                runtime
                    .execution()
                    .complete_future(&handle, stored)
                    .map_err(execution_error)?;
                return result
                    .map(|value| AwaitCompletion { edge, value })
                    .map_err(NativeExecutorError::from);
            }
        }
    }
}

fn normalize_outputs(value: Value, requested_outputs: usize) -> Value {
    match value {
        Value::OutputList(mut values) if requested_outputs == 1 && values.len() == 1 => {
            values.remove(0)
        }
        value => value,
    }
}

fn execution_error(error: ExecutionServiceError) -> NativeExecutorError {
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
