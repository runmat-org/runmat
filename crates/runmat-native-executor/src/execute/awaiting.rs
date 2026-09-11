use runmat_native_codegen::NativeEdge;
use runmat_runtime::execution::{AwaitAction, DeferredCall, ExecutionServiceError};
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_value::{Value, ValueSequence};

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
        .begin_await(value)
        .map_err(execution_error)?;
    match action {
        AwaitAction::Passthrough(value) => Ok(AwaitStart::Ready(value)),
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
            let (continuation, generation) = state.next_suspension_identity()?;
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
                    AwaitAction::Passthrough(value) => return Ok(AwaitCompletion { edge, value }),
                    AwaitAction::Completed(sequence) => {
                        return Ok(AwaitCompletion {
                            edge,
                            value: resolve_await_value(sequence)?,
                        });
                    }
                    AwaitAction::Pending(value) => work = AwaitWork::Poll(value),
                    AwaitAction::ExecuteFuture { handle, call } => {
                        work = AwaitWork::Execute { handle, call };
                    }
                }
            }
            AwaitWork::Execute { handle, call } => {
                let requested_outputs = call.invocation.requested_outputs();
                let result = match call.invocation {
                    runmat_runtime::execution::DeferredInvocation::Callable(descriptor) => {
                        runtime
                            .scope(
                                runmat_runtime::call::descriptor::execute_callable_descriptor(
                                    descriptor,
                                ),
                            )
                            .await
                    }
                    runmat_runtime::execution::DeferredInvocation::Program { .. } => {
                        Err(runmat_runtime::runtime_error::semantic_error(
                            "ParallelProgramUnavailable",
                            "native continuation cannot execute a deferred program in-process",
                        ))
                    }
                }
                .map_err(NativeExecutorError::from)
                .and_then(|sequence| normalize_outputs(sequence, requested_outputs));
                let stored = result
                    .as_ref()
                    .map(Clone::clone)
                    .map_err(|error| ExecutionServiceError::Failed(error.to_string()));
                runtime
                    .execution()
                    .complete_future(&handle, stored)
                    .map_err(execution_error)?;
                let sequence = result?;
                return Ok(AwaitCompletion {
                    edge,
                    value: resolve_await_value(sequence)?,
                });
            }
        }
    }
}

fn normalize_outputs(
    sequence: ValueSequence,
    requested_outputs: usize,
) -> NativeExecutorResult<ValueSequence> {
    runmat_value::validate_output_count(requested_outputs)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
    if requested_outputs == 0 {
        return Ok(ValueSequence::empty());
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

fn resolve_await_value(sequence: ValueSequence) -> Result<Value, NativeExecutorError> {
    sequence
        .resolve(
            runmat_types::SequenceUse::RequireSingle,
            runmat_runtime::sequence::SequenceResolutionContext::default(),
        )
        .map(|mut values| values.remove(0))
        .map_err(NativeExecutorError::from)
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
