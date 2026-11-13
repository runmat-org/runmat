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
use futures::{FutureExt, TryFutureExt};
use runmat_value::{BoundFunctionBinding, Value};
use std::future::Future;

type ExecutionFuture<'a, T> = crate::execution::RuntimeContinuationFuture<'a, T>;
pub(super) fn admit_future<'a, F, O>(
    runtime: Option<&RuntimeContext>,
    future: F,
) -> ExecutionFuture<'a, O>
where
    F: Future<Output = Result<O, RuntimeError>> + 'a,
    O: crate::execution::rooted_continuation::RootedCompletion + 'a,
{
    crate::execution::rooted_continuation::admit_rooted_future(
        runtime,
        "callable descriptor",
        crate::builtin::builtin_control_error,
        future,
    )
}
pub(super) trait DescriptorArguments {
    fn prepare(
        self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<crate::call::BuiltinArguments, RuntimeError>;
}
impl DescriptorArguments for Vec<Value> {
    fn prepare(
        self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<crate::call::BuiltinArguments, RuntimeError> {
        match runtime {
            Some(runtime) => RootedValues::in_context(self, runtime),
            None => RootedValues::new(self).map_err(|e| e.into_runtime_error()),
        }
        .map(Into::into)
    }
}
impl DescriptorArguments for crate::call::BuiltinArguments {
    fn prepare(self, _runtime: Option<&RuntimeContext>) -> Result<Self, RuntimeError> {
        Ok(self)
    }
}
fn root_arguments(
    runtime: Option<&RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> Result<crate::call::RootedFunctionArguments, RuntimeError> {
    args.into_supplied_function_arguments(runtime)
}

#[inline(never)]
fn captured_resolved(
    runtime: Option<&RuntimeContext>,
    identity: crate::call::identity::CallableIdentityOwner,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    closure: super::RetainedClosureValue,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    let callable = match closure.retain_for_call(runtime) {
        Ok(callable) => callable,
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    let supplied = match root_arguments(runtime, args) {
        Ok(supplied) => supplied,
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    let arguments = match crate::call::BuiltinArguments::new(supplied)
        .rebind_callee(Some(callable), runtime)
        .map(|args| args.into_function_arguments())
    {
        Ok(arguments) => arguments,
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    admit_future(runtime, async move {
        if let runmat_types::CallableIdentity::BoundFunction(function) = identity.as_identity() {
            let Some(result) = crate::callsite::with_function_input_callsite(
                metadata.function_input_callsite.clone(),
                crate::user_functions::try_call_rooted_semantic_function_in_context(
                    runtime,
                    function.0,
                    arguments,
                    requested_outputs,
                ),
            )
            .await?
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

pub(super) fn execute<A: DescriptorArguments>(
    runtime: Option<&RuntimeContext>,
    descriptor: CallableDescriptor<A>,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + '_ {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    let selected = match args.prepare(runtime) {
        Err(error) => admit_future(runtime, async move { Err(error) }),
        Ok(args) => match target {
            CallableTarget::Scoped(handle) => {
                scoped(runtime, handle, args, requested_outputs, metadata)
            }
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
            CallableTarget::FevalForward(value) => forwarded(
                runtime,
                value,
                args,
                requested_outputs,
                metadata.function_input_callsite,
            ),
            CallableTarget::RetainedFevalForward(value) => admit_future(
                runtime,
                invoke_retained_forwarded(runtime, value, args, requested_outputs, metadata),
            ),
            CallableTarget::RetainedBuiltin(value) => admit_future(
                runtime,
                invoke_retained_builtin(runtime, value, args, requested_outputs, metadata),
            ),
        },
    };
    selected.map_ok(move |value| {
        if requested_outputs == 0 {
            RootedValueSequence::empty()
        } else {
            value
        }
    })
}

pub(super) fn try_execute<A: DescriptorArguments>(
    runtime: Option<&RuntimeContext>,
    descriptor: CallableDescriptor<A>,
) -> impl Future<Output = Result<Option<RootedValueSequence>, RuntimeError>> + '_ {
    let CallableDescriptor {
        target,
        args,
        requested_outputs,
        metadata,
    } = descriptor;
    let selected = match args.prepare(runtime) {
        Err(error) => admit_future(runtime, async move { Err(error) }),
        Ok(args) => match target {
            CallableTarget::Scoped(handle) => admit_future(
                runtime,
                scoped(runtime, handle, args, requested_outputs, metadata).map_ok(Some),
            ),
            CallableTarget::Bound(binding) => {
                try_bound(runtime, binding, args, requested_outputs, metadata)
            }
            CallableTarget::BoundLexical { binding, closure } => admit_future(
                runtime,
                bound_lexical(runtime, binding, closure, args, requested_outputs, metadata)
                    .map_ok(Some),
            ),
            CallableTarget::CapturedResolved {
                identity,
                fallback_policy,
                closure,
            } => admit_future(
                runtime,
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
            CallableTarget::FevalForward(value) => try_forwarded(
                runtime,
                value,
                args,
                requested_outputs,
                metadata.function_input_callsite,
            ),
            CallableTarget::RetainedFevalForward(value) => admit_future(
                runtime,
                invoke_retained_forwarded(runtime, value, args, requested_outputs, metadata)
                    .map_ok(Some),
            ),
            CallableTarget::RetainedBuiltin(value) => admit_future(
                runtime,
                invoke_retained_builtin(runtime, value, args, requested_outputs, metadata)
                    .map_ok(Some),
            ),
        },
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
fn scoped(
    runtime: Option<&RuntimeContext>,
    handle: runmat_value::ScopedFunctionHandle,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'static, RootedValueSequence> {
    crate::execution::scoped_callable::invoke(
        runtime,
        handle,
        args.into_function_arguments(),
        requested_outputs,
        metadata.function_input_callsite,
    )
    .unwrap_or_else(crate::execution::RuntimeContinuationFuture::rejected)
}

#[inline(never)]
fn bound(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    let arguments = match root_arguments(runtime, args) {
        Ok(arguments) => arguments,
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    admit_future(
        runtime,
        invoke_bound(
            runtime,
            binding,
            arguments,
            requested_outputs,
            metadata,
            None,
        ),
    )
}

#[inline(never)]
fn bound_lexical(
    runtime: Option<&RuntimeContext>,
    binding: BoundFunctionBinding,
    closure: super::RetainedClosureValue,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    if closure.as_closure().binding.as_ref() != Some(&binding) {
        return admit_future(runtime, async {
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
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    let environment = closure.as_closure().lexical_environment.clone();
    let supplied = match root_arguments(runtime, args) {
        Ok(supplied) => supplied,
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    let arguments = match crate::call::BuiltinArguments::new(supplied)
        .rebind_callee(Some(callable), runtime)
        .map(|args| args.into_function_arguments())
    {
        Ok(arguments) => arguments,
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    admit_future(runtime, async move {
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
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    let arguments = match root_arguments(runtime, args) {
        Ok(arguments) => arguments,
        Err(error) => return admit_future(runtime, async move { Err(error) }),
    };
    admit_future(
        runtime,
        invoke_bound(
            runtime,
            binding,
            arguments,
            requested_outputs,
            metadata,
            None,
        )
        .map_ok(Some),
    )
}

pub(super) fn invoke_bound<'a>(
    runtime: Option<&'a RuntimeContext>,
    binding: BoundFunctionBinding,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    let function = binding.function;
    try_invoke_bound(
        runtime,
        binding,
        args,
        requested_outputs,
        metadata.clone(),
        lexical_environment,
    )
    .map(move |result| result?.ok_or_else(|| function_unavailable_error(function, &metadata)))
}

fn try_invoke_bound<'a>(
    runtime: Option<&'a RuntimeContext>,
    binding: BoundFunctionBinding,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> impl Future<Output = Result<Option<RootedValueSequence>, RuntimeError>> + 'a {
    crate::callsite::with_function_input_callsite(
        metadata.function_input_callsite.clone(),
        async move {
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
        },
    )
    .map(move |result| {
        drop(metadata);
        result?.transpose()
    })
}

#[inline(never)]
fn resolved(
    runtime: Option<&RuntimeContext>,
    identity: crate::call::identity::CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
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
    identity: crate::call::identity::CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
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
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> ExecutionFuture<'_, RootedValueSequence> {
    admit_future(
        runtime,
        invoke_forwarded(runtime, value, args, requested_outputs, input_callsite),
    )
}

#[inline(never)]
fn try_forwarded(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    admit_future(
        runtime,
        invoke_forwarded(runtime, value, args, requested_outputs, input_callsite).map_ok(Some),
    )
}

async fn invoke_forwarded(
    runtime: Option<&RuntimeContext>,
    value: Value,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<RootedValueSequence, RuntimeError> {
    crate::call::feval::invoke_retained(runtime, value, args, requested_outputs, input_callsite)
        .await
}

async fn invoke_retained_forwarded(
    runtime: Option<&RuntimeContext>,
    value: super::RetainedCallableValue,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> Result<RootedValueSequence, RuntimeError> {
    let name = super::repeatable::retained_target_name(value.value());
    let Some(name) = name else {
        return crate::call::feval::invoke_object_retained(
            runtime,
            value.value().clone(),
            args,
            requested_outputs,
        )
        .await;
    };
    let captures = matches!(value.value(), Value::Closure(_));
    let args = args
        .rebind_callee(captures.then(|| value.0.clone()), runtime)?
        .into_function_arguments();
    match crate::user_functions::try_call_rooted_semantic_name_in_context(
        runtime,
        name,
        args,
        requested_outputs,
    )
    .await
    {
        std::ops::ControlFlow::Break(result) => result,
        std::ops::ControlFlow::Continue(args) => {
            let (identity, fallback) = match value.value() {
                Value::ExternalFunctionHandle(name) => (
                    crate::external_callable_identity_for_name(name),
                    runmat_types::CallableFallbackPolicy::ExternalBoundary,
                ),
                Value::MethodFunctionHandle(name) => (
                    runmat_types::CallableIdentity::Method(runmat_types::MethodId(
                        name.name().to_owned(),
                    )),
                    runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
                ),
                _ => crate::callable_identity_for_handle_name(name),
            };
            crate::dispatch_prepared_callable_with_policy_in_runtime(
                runtime,
                identity,
                fallback,
                args,
                requested_outputs,
                metadata.function_input_callsite,
            )
            .await
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(runtime) = runtime {
            let selected = runtime.clone();
            let environment = closure.lexical_environment.clone();
            // A mapped callback may enter another interpreter from an active
            // builtin. Keep that owned child on the existing host supervisor,
            // not on the suspended mapping operation's poll stack.
            return crate::context::run_selected_callback_continuation_with(
                runtime,
                move || async move {
                    try_invoke_bound(
                        Some(&selected),
                        binding,
                        args,
                        requested_outputs,
                        metadata,
                        environment,
                    )
                    .await
                },
            )
            .await;
        }
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

/// Keep the admitted source generation alive while dispatch borrows its name.
/// In particular this path never performs semantic name lookup again.
async fn invoke_retained_builtin(
    runtime: Option<&RuntimeContext>,
    value: super::RetainedCallableValue,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> Result<RootedValueSequence, RuntimeError> {
    let name = super::named_selection::retained_name(value.value()).ok_or_else(|| {
        crate::runtime_error::semantic_error(
            "CallableTarget",
            "selected builtin has no retained name",
        )
    })?;
    super::call_builtin_with_requested_outputs(
        runtime,
        name,
        args,
        requested_outputs,
        metadata.function_input_callsite,
    )
    .await
}
