//! Heap ownership for one already-classified callable identity family.
//!
//! Identity selection is synchronous and effect-free. Each constructor is a
//! non-inlined boundary that allocates only the selected family's cold future;
//! unrelated semantic, external, builtin, and fallback futures never become
//! variants of one native poll frame.

use super::{
    call_builtin_with_requested_outputs, forward_named_fallback, function_unavailable_error,
    undefined_identity_error, CallableMetadata,
};
use crate::call::identity::CallableIdentityOwner;
use crate::context::RuntimeContext;
use crate::execution::RootedValueSequence;
use crate::RuntimeError;
use runmat_types::{CallableFallbackPolicy, CallableIdentity, FunctionId};

use super::execution::admit_future;

type ExecutionFuture<'a, T> = crate::execution::RuntimeContinuationFuture<'a, T>;

pub(super) fn execute(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, RootedValueSequence> {
    match identity.as_identity() {
        CallableIdentity::Builtin(_) => {
            builtin(runtime, identity, args, requested_outputs, metadata)
        }
        CallableIdentity::BoundFunction(_) => {
            semantic(runtime, identity, args, requested_outputs, metadata)
        }
        CallableIdentity::ExternalFunction { .. } => {
            external(runtime, identity, args, requested_outputs, metadata)
        }
        _ => descriptor(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata,
            fallback_policy,
        ),
    }
}

pub(super) fn try_execute(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    match identity.as_identity() {
        CallableIdentity::Builtin(_) => {
            try_builtin(runtime, identity, args, requested_outputs, metadata)
        }
        CallableIdentity::BoundFunction(_) => {
            try_semantic(runtime, identity, args, requested_outputs, metadata)
        }
        CallableIdentity::ExternalFunction { .. } => {
            try_external(runtime, identity, args, requested_outputs, metadata)
        }
        _ => try_descriptor(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata,
            fallback_policy,
        ),
    }
}

#[inline(never)]
fn builtin(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    admit_future(
        runtime,
        invoke_builtin(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata.function_input_callsite,
        ),
    )
}

#[inline(never)]
fn try_builtin(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    admit_future(runtime, async move {
        invoke_builtin(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata.function_input_callsite,
        )
        .await
        .map(Some)
    })
}

async fn invoke_builtin(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<RootedValueSequence, RuntimeError> {
    let CallableIdentity::Builtin(id) = identity.as_identity() else {
        unreachable!("classified builtin family")
    };
    let result = call_builtin_with_requested_outputs(
        runtime,
        &id.0,
        args,
        requested_outputs,
        input_callsite,
    )
    .await;
    drop(identity);
    result
}

#[inline(never)]
fn semantic(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    admit_future(runtime, async move {
        let CallableIdentity::BoundFunction(function) = identity.as_identity() else {
            unreachable!("classified semantic family")
        };
        let function = *function;
        match invoke_semantic(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata.function_input_callsite.clone(),
        )
        .await?
        {
            Some(value) => Ok(value),
            None => Err(function_unavailable_error(function.0, &metadata)),
        }
    })
}

#[inline(never)]
fn try_semantic(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    admit_future(
        runtime,
        invoke_semantic(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata.function_input_callsite,
        ),
    )
}

async fn invoke_semantic(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let CallableIdentity::BoundFunction(function) = identity.as_identity() else {
        unreachable!("classified semantic family")
    };
    let result = match crate::callsite::with_function_input_callsite(
        input_callsite,
        crate::user_functions::try_call_rooted_semantic_function_in_context(
            runtime,
            function.0,
            args.into_supplied_function_arguments(runtime)?,
            requested_outputs,
        ),
    )
    .await?
    {
        Some(result) => result.map(Some),
        None => Ok(None),
    };
    drop(identity);
    result
}

#[inline(never)]
fn external(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    admit_future(runtime, async move {
        let (function, display_name, retained_identity) = external_fields(identity);
        let args = args.into_supplied_function_arguments(runtime)?;
        let result = match invoke_external(
            runtime,
            function,
            display_name,
            args,
            requested_outputs,
            metadata.function_input_callsite.clone(),
        )
        .await?
        {
            Some(value) => Ok(value),
            None => Err(function_unavailable_error(function.0, &metadata)),
        };
        drop(retained_identity);
        result
    })
}

#[inline(never)]
fn try_external(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    admit_future(runtime, async move {
        let (function, display_name, retained_identity) = external_fields(identity);
        let args = args.into_supplied_function_arguments(runtime)?;
        let result = invoke_external(
            runtime,
            function,
            display_name,
            args,
            requested_outputs,
            metadata.function_input_callsite,
        )
        .await;
        drop(retained_identity);
        result
    })
}

fn external_fields(
    identity: CallableIdentityOwner,
) -> (FunctionId, String, Option<std::sync::Arc<CallableIdentity>>) {
    match identity {
        CallableIdentityOwner::Owned(CallableIdentity::ExternalFunction {
            function,
            display_name,
        }) => (function, display_name, None),
        CallableIdentityOwner::Retained(owner) => {
            let CallableIdentity::ExternalFunction {
                function,
                display_name,
            } = owner.as_ref()
            else {
                unreachable!("classified external family")
            };
            // ExternalFunctionCall still owns its presentation String. This
            // existing adapter boundary is not a copy of semantic identity or
            // QName segments; the original source owner survives the await.
            (*function, display_name.clone(), Some(owner))
        }
        _ => unreachable!("classified external family"),
    }
}

async fn invoke_external(
    runtime: Option<&RuntimeContext>,
    function: FunctionId,
    display_name: String,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let call = crate::user_functions::ExternalFunctionCall {
        function: function.0,
        display_name,
        arguments: args,
        requested_outputs,
    };
    let call = match crate::callsite::with_function_input_callsite(
        input_callsite.clone(),
        crate::user_functions::try_call_external_function_owned_in_context(runtime, call),
    )
    .await?
    {
        std::ops::ControlFlow::Break(result) => return result.map(Some),
        std::ops::ControlFlow::Continue(call) => call,
    };
    if crate::user_functions::current_semantic_function_invoker().is_none() {
        return Ok(None);
    }
    match crate::callsite::with_function_input_callsite(
        input_callsite,
        crate::user_functions::try_call_rooted_semantic_function_in_context(
            runtime,
            function.0,
            call.arguments,
            requested_outputs,
        ),
    )
    .await?
    {
        Some(result) => result.map(Some),
        None => Ok(None),
    }
}

#[inline(never)]
fn descriptor(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, RootedValueSequence> {
    admit_future(runtime, async move {
        let args = args.into_supplied_function_arguments(runtime)?;
        invoke_descriptor_required(
            runtime,
            identity,
            args,
            requested_outputs,
            &metadata,
            fallback_policy,
        )
        .await
    })
}

#[inline(never)]
fn try_descriptor(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    admit_future(runtime, async move {
        let args = args.into_supplied_function_arguments(runtime)?;
        invoke_descriptor_optional(
            runtime,
            identity,
            args,
            requested_outputs,
            metadata.function_input_callsite,
            fallback_policy,
        )
        .await
    })
}

pub(super) async fn invoke_descriptor_required(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    mut args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    metadata: &CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    if let CallableIdentity::Builtin(id) = identity.as_identity() {
        return super::forward_named_fallback(
            runtime,
            id.0.clone(),
            crate::call::BuiltinArguments::new(args),
            requested_outputs,
            metadata.function_input_callsite.clone(),
            metadata.source_ref,
            metadata.source_invocation,
        )
        .await;
    }
    let request = crate::user_functions::CallableRequest::rooted_resolved_with_identity_owner(
        identity.clone(),
        fallback_policy,
        args,
        requested_outputs,
    )
    .with_function_input_callsite(metadata.function_input_callsite.clone())
    .with_selected_runtime(runtime);
    match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
        std::ops::ControlFlow::Break(result) => return result,
        std::ops::ControlFlow::Continue(returned) => args = returned,
    }
    forward_descriptor_fallback(
        runtime,
        identity.as_identity(),
        args,
        requested_outputs,
        metadata,
        fallback_policy,
    )
    .await
}

/// Continue only after semantic selection declined the exact original owner.
/// Repeated callbacks use this port rather than replaying semantic dispatch.
pub(super) async fn forward_descriptor_fallback(
    runtime: Option<&RuntimeContext>,
    identity: &CallableIdentity,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    metadata: &CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    let name = match identity {
        CallableIdentity::Builtin(id) => Some(id.0.clone()),
        _ => fallback_policy.vm_fallback_name_for(identity),
    };
    let Some(name) = name else {
        return Err(undefined_identity_error(identity, metadata));
    };
    forward_named_fallback(
        runtime,
        name,
        crate::call::BuiltinArguments::new(args),
        requested_outputs,
        metadata.function_input_callsite.clone(),
        metadata.source_ref,
        metadata.source_invocation,
    )
    .await
}

async fn invoke_descriptor_optional(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentityOwner,
    mut args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    if let CallableIdentity::Builtin(id) = identity.as_identity() {
        return crate::callsite::with_callsite(
            input_callsite,
            crate::dispatcher::try_call_retained_builtin_sequence_with_outputs(
                &id.0,
                crate::call::BuiltinArguments::new(args),
                requested_outputs,
                runtime,
            ),
        )
        .await?;
    }
    let request = crate::user_functions::CallableRequest::rooted_resolved_with_identity_owner(
        identity.clone(),
        fallback_policy,
        args,
        requested_outputs,
    )
    .with_function_input_callsite(input_callsite.clone())
    .with_selected_runtime(runtime);
    match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
        std::ops::ControlFlow::Break(result) => return result.map(Some),
        std::ops::ControlFlow::Continue(returned) => args = returned,
    }
    let Some(name) = fallback_policy.vm_fallback_name_for(identity.as_identity()) else {
        return Ok(None);
    };
    let invoke = crate::callsite::with_callsite(
        input_callsite,
        crate::dispatcher::try_call_retained_builtin_sequence_with_outputs(
            &name,
            crate::call::BuiltinArguments::new(args),
            requested_outputs,
            runtime,
        ),
    );
    let result = match runtime {
        Some(runtime) => runtime.scope(invoke).await,
        None => invoke.await,
    }??;
    Ok(result)
}
