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
use crate::execution::{RootedValueSequence, RootedValues};
use crate::RuntimeError;
use futures::TryFutureExt;
use runmat_value::{BoundFunctionBinding, Value};
use std::future::Future;
use std::pin::Pin;

type ExecutionFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, RuntimeError>> + 'a>>;

fn root_arguments(
    runtime: Option<&RuntimeContext>,
    args: Vec<Value>,
) -> Result<RootedValues, RuntimeError> {
    match runtime {
        Some(runtime) => RootedValues::in_context(args, runtime),
        None => RootedValues::new(args).map_err(|error| error.into_runtime_error()),
    }
}

/// Invoke one bound closure without flattening its immutable capture
/// generation into the explicit argument suffix. Callers that retain a
/// callable across repeated invocations can clone `RetainedClosureValue`
/// in O(1) and keep the same positional capture owner.
pub(crate) async fn try_execute_retained_bound_closure_in_context(
    runtime: Option<&RuntimeContext>,
    closure: super::RetainedClosureValue,
    supplied: Vec<Value>,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let Some(binding) = closure.as_closure().binding.clone() else {
        return Ok(None);
    };
    let lexical_environment = closure.as_closure().lexical_environment.clone();
    let selected_runtime = runtime.cloned().or_else(|| {
        lexical_environment
            .as_ref()
            .and_then(|_| crate::context::legacy::active())
    });
    let callable = closure.retain_for_call(selected_runtime.as_ref())?;
    let supplied = root_arguments(selected_runtime.as_ref(), supplied)?;
    let arguments = crate::call::RootedFunctionArguments::captured(callable, supplied)?;
    match selected_runtime.as_ref() {
        Some(runtime) => crate::callsite::with_function_input_callsite(
            input_callsite,
            crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                runtime,
                binding,
                arguments,
                requested_outputs,
                lexical_environment,
            ),
        )
        .await?
        .transpose(),
        None if lexical_environment.is_some() => Err(crate::runtime_error::semantic_error(
            "LexicalEnvironment",
            "nested closure requires an explicit runtime context",
        )),
        None => crate::callsite::with_function_input_callsite(
            input_callsite,
            crate::user_functions::try_call_rooted_bound_function(
                binding,
                arguments,
                requested_outputs,
            ),
        )
        .await?
        .transpose(),
    }
}

#[inline(never)]
fn captured_resolved(
    runtime: Option<&RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    closure: super::RetainedClosureValue,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    let callable = match closure.retain_for_call(runtime) {
        Ok(callable) => callable,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    let supplied = match root_arguments(runtime, args) {
        Ok(supplied) => supplied,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    let arguments = match crate::call::RootedFunctionArguments::captured(callable, supplied) {
        Ok(arguments) => arguments,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    // The unbound semantic/builtin/external port still requires a contiguous
    // Value slice. Materialization is an explicit legacy boundary, denied
    // before any effects when its physical backing is not finite-admitted.
    let flattened = match arguments.materialize_legacy_contiguous(runtime) {
        Ok(flattened) => flattened,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    Box::pin(async move {
        let _source_roots = arguments;
        resolved_execution::execute(
            runtime,
            identity,
            flattened,
            requested_outputs,
            metadata,
            fallback_policy,
        )
        .await
    })
}

pub(super) fn execute(
    runtime: Option<&RuntimeContext>,
    descriptor: CallableDescriptor,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + '_ {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    let selected = match target {
        CallableTarget::Bound(binding) => {
            bound(runtime, binding, args, requested_outputs, metadata)
        }
        CallableTarget::BoundLexical { binding, closure } => {
            bound_lexical(runtime, binding, closure, args, requested_outputs, metadata)
        }
        CallableTarget::CapturedResolved {
            identity,
            fallback_policy,
            closure,
        } => captured_resolved(
            runtime,
            identity,
            fallback_policy,
            closure,
            args,
            requested_outputs,
            metadata,
        ),
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
    };
    selected.map_ok(move |value| {
        if requested_outputs == 0 {
            RootedValueSequence::empty()
        } else {
            value
        }
    })
}

pub(super) fn try_execute(
    runtime: Option<&RuntimeContext>,
    descriptor: CallableDescriptor,
) -> impl Future<Output = Result<Option<RootedValueSequence>, RuntimeError>> + '_ {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    let selected = match target {
        CallableTarget::Bound(binding) => {
            try_bound(runtime, binding, args, requested_outputs, metadata)
        }
        CallableTarget::BoundLexical { binding, closure } => Box::pin(
            bound_lexical(runtime, binding, closure, args, requested_outputs, metadata)
                .map_ok(Some),
        ),
        CallableTarget::CapturedResolved {
            identity,
            fallback_policy,
            closure,
        } => Box::pin(
            captured_resolved(
                runtime,
                identity,
                fallback_policy,
                closure,
                args,
                requested_outputs,
                metadata,
            )
            .map_ok(Some),
        ),
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
    };
    selected.map_ok(move |value| {
        value.map(|value| {
            if requested_outputs == 0 {
                RootedValueSequence::empty()
            } else {
                value
            }
        })
    })
}

#[inline(never)]
fn bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    let arguments = match root_arguments(runtime, args) {
        Ok(arguments) => arguments,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    Box::pin(invoke_bound(
        runtime,
        binding,
        crate::call::RootedFunctionArguments::ordinary(arguments),
        requested_outputs,
        metadata,
        None,
    ))
}

#[inline(never)]
fn bound_lexical(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    closure: super::RetainedClosureValue,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    if closure.as_closure().binding.as_ref() != Some(&binding) {
        return Box::pin(async {
            Err(crate::runtime_error::semantic_error(
                "UnavailableBoundFunction",
                "captured closure disagrees with its selected bound function",
            ))
        });
    }
    // The descriptor stores a root-free shared capture generation. Admit its
    // external root lease now, before an unpolled future can enter a queue.
    let callable = match closure.retain_for_call(runtime) {
        Ok(callable) => callable,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    let environment = closure.as_closure().lexical_environment.clone();
    let supplied = match root_arguments(runtime, args) {
        Ok(supplied) => supplied,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    let arguments = match crate::call::RootedFunctionArguments::captured(callable, supplied) {
        Ok(arguments) => arguments,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    Box::pin(async move {
        invoke_bound(
            runtime,
            binding,
            arguments,
            requested_outputs,
            metadata,
            environment,
        )
        .await
    })
}

#[inline(never)]
fn try_bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    let arguments = match root_arguments(runtime, args) {
        Ok(arguments) => arguments,
        Err(error) => return Box::pin(async move { Err(error) }),
    };
    Box::pin(
        invoke_bound(
            runtime,
            binding,
            crate::call::RootedFunctionArguments::ordinary(arguments),
            requested_outputs,
            metadata,
            None,
        )
        .map_ok(Some),
    )
}

async fn invoke_bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<RootedValueSequence, RuntimeError> {
    let function = binding.function;
    crate::callsite::with_function_input_callsite(metadata.function_input_callsite.clone(), async {
        match runtime {
            Some(runtime) => {
                crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                    runtime,
                    binding,
                    args,
                    requested_outputs,
                    lexical_environment,
                )
                .await
            }
            None => {
                if lexical_environment.is_some() {
                    return Some(Err(crate::runtime_error::semantic_error(
                        "LexicalEnvironment",
                        "nested closure requires an explicit runtime context",
                    )));
                }
                crate::user_functions::try_call_rooted_bound_function(
                    binding,
                    args,
                    requested_outputs,
                )
                .await
            }
        }
    })
    .await?
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
) -> ExecutionFuture<'_, RootedValueSequence> {
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
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
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
) -> ExecutionFuture<'_, RootedValueSequence> {
    Box::pin(invoke_forwarded(runtime, value, args, requested_outputs))
}

#[inline(never)]
fn try_forwarded(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    Box::pin(invoke_forwarded(runtime, value, args, requested_outputs).map_ok(Some))
}

async fn invoke_forwarded(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
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
