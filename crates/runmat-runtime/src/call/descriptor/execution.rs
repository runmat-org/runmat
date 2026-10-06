//! One heap-owned future for the selected callable target.
//!
//! Selection is synchronous and effect-free. Each non-inlined constructor owns
//! only its target's future, so polling a bound semantic call does not reserve
//! native stack space for name resolution or forwarded feval execution.

use super::{
    function_unavailable_error, resolved_execution, CallableDescriptor, CallableMetadata,
    CallableTarget,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;
use futures::TryFutureExt;
use runmat_value::{BoundFunctionBinding, Value, ValueSequence};
use std::future::Future;
use std::pin::Pin;

type ExecutionFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, RuntimeError>> + 'a>>;

pub(super) fn execute(
    runtime: Option<&RuntimeContext>,
    descriptor: CallableDescriptor,
) -> ExecutionFuture<'_, ValueSequence> {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    match target {
        CallableTarget::Bound(binding) => {
            bound(runtime, binding, args, requested_outputs, metadata)
        }
        CallableTarget::Resolved {
            identity,
            fallback_policy,
        } => resolved(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata,
            fallback_policy,
        ),
        CallableTarget::FevalForward(value) => forwarded(runtime, value, args, requested_outputs),
    }
}

pub(super) fn try_execute(
    runtime: Option<&RuntimeContext>,
    descriptor: CallableDescriptor,
) -> ExecutionFuture<'_, Option<ValueSequence>> {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    match target {
        CallableTarget::Bound(binding) => {
            try_bound(runtime, binding, args, requested_outputs, metadata)
        }
        CallableTarget::Resolved {
            identity,
            fallback_policy,
        } => try_resolved(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata,
            fallback_policy,
        ),
        CallableTarget::FevalForward(value) => {
            try_forwarded(runtime, value, args, requested_outputs)
        }
    }
}

#[inline(never)]
fn bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, ValueSequence> {
    Box::pin(invoke_bound(
        runtime,
        binding,
        args,
        requested_outputs,
        metadata,
    ))
}

#[inline(never)]
fn try_bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<ValueSequence>> {
    Box::pin(invoke_bound(runtime, binding, args, requested_outputs, metadata).map_ok(Some))
}

async fn invoke_bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> Result<ValueSequence, RuntimeError> {
    let function = binding.function;
    crate::callsite::with_function_input_callsite(metadata.function_input_callsite.clone(), async {
        match runtime {
            Some(runtime) => {
                crate::user_functions::try_call_bound_function_in_context(
                    runtime,
                    binding,
                    &args,
                    requested_outputs,
                )
                .await
            }
            None => {
                crate::user_functions::try_call_bound_function(binding, &args, requested_outputs)
                    .await
            }
        }
    })
    .await
    .unwrap_or_else(|| Err(function_unavailable_error(function, &metadata)))
}

#[inline(never)]
fn resolved(
    runtime: Option<&RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: runmat_types::CallableFallbackPolicy,
) -> ExecutionFuture<'_, ValueSequence> {
    resolved_execution::execute(
        runtime,
        identity,
        args,
        requested_outputs,
        metadata,
        fallback_policy,
    )
}

#[inline(never)]
fn try_resolved(
    runtime: Option<&RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: runmat_types::CallableFallbackPolicy,
) -> ExecutionFuture<'_, Option<ValueSequence>> {
    resolved_execution::try_execute(
        runtime,
        identity,
        args,
        requested_outputs,
        metadata,
        fallback_policy,
    )
}

#[inline(never)]
fn forwarded(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> ExecutionFuture<'_, ValueSequence> {
    Box::pin(invoke_forwarded(runtime, value, args, requested_outputs))
}

#[inline(never)]
fn try_forwarded(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> ExecutionFuture<'_, Option<ValueSequence>> {
    Box::pin(invoke_forwarded(runtime, value, args, requested_outputs).map_ok(Some))
}

async fn invoke_forwarded(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<ValueSequence, RuntimeError> {
    match runtime {
        Some(runtime) => {
            crate::call_feval_sequence_async_with_outputs_in_context(
                runtime,
                value,
                &args,
                requested_outputs,
            )
            .await
        }
        None => {
            crate::call_feval_sequence_async_with_outputs(value, &args, requested_outputs).await
        }
    }
}
