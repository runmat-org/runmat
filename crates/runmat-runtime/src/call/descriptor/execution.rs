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
    Box::pin(async move {
        if let runmat_types::CallableIdentity::BoundFunction(function) = identity {
            let Some(result) = crate::user_functions::try_call_rooted_semantic_function_in_context(
                runtime,
                function.0,
                arguments,
                requested_outputs,
            )
            .await
            else {
                return Err(function_unavailable_error(function.0, &metadata));
            };
            return result;
        }
        // The same selected-callee continuation handles both ordinary and
        // managed calls. Semantic decline forwards the original capture owner,
        // including its callsite/source metadata, to canonical builtin dispatch.
        resolved_execution::invoke_descriptor_required(
            runtime,
            identity,
            arguments,
            requested_outputs,
            &metadata,
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
        CallableTarget::RetainedFevalForward(_) => Box::pin(async move {
            Err(crate::runtime_error::semantic_error(
                "RunMat:Callable:RepeatableOwner",
                "retained forwarded target requires its repeatable root owner",
            ))
        }),
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
        CallableTarget::RetainedFevalForward(_) => Box::pin(async move {
            Err(crate::runtime_error::semantic_error(
                "RunMat:Callable:RepeatableOwner",
                "retained forwarded target requires its repeatable root owner",
            ))
        }),
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

pub(super) async fn invoke_bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<RootedValueSequence, RuntimeError> {
    let function = binding.function;
    try_invoke_bound(
        runtime,
        binding,
        args,
        requested_outputs,
        metadata.clone(),
        lexical_environment,
    )
    .await?
    .ok_or_else(|| function_unavailable_error(function, &metadata))
}

async fn try_invoke_bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
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
    .transpose()
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

/// Invoke the already-selected closure with its original capture generation.
/// This does not resolve its name again or normalize true captures as supplied.
pub(crate) async fn invoke_selected_retained_closure(
    runtime: Option<&RuntimeContext>,
    closure: &super::RetainedClosureValue,
    callable: crate::call::RetainedCallableTarget,
    arguments: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let args = arguments
        .rebind_callee(Some(callable), runtime)?
        .into_function_arguments();
    let closure = closure.as_closure();
    let metadata = CallableMetadata::feval(Some(closure.function_name.clone()));
    if let Some(binding) = closure.binding.clone() {
        return try_invoke_bound(
            runtime,
            binding,
            args,
            requested_outputs,
            metadata,
            closure.lexical_environment.clone(),
        )
        .await;
    }
    crate::dispatch_prepared_callable_with_policy_in_runtime(
        runtime,
        runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
            closure.function_name.to_string(),
        )),
        runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
        args,
        requested_outputs,
        metadata.function_input_callsite,
    )
    .await
    .map(Some)
}
