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
use crate::context::RuntimeContext;
use crate::execution::RootedValueSequence;
use crate::RuntimeError;
use runmat_types::{BuiltinId, CallableFallbackPolicy, CallableIdentity, FunctionId};
use runmat_value::Value;
use std::future::Future;
use std::pin::Pin;

type ExecutionFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, RuntimeError>> + 'a>>;

pub(super) fn execute(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, RootedValueSequence> {
    match identity {
        CallableIdentity::Builtin(id) => builtin(runtime, id, args, requested_outputs, metadata),
        CallableIdentity::BoundFunction(function) => {
            semantic(runtime, function, args, requested_outputs, metadata)
        }
        CallableIdentity::ExternalFunction {
            function,
            display_name,
        } => external(
            runtime,
            function,
            display_name,
            args,
            requested_outputs,
            metadata,
        ),
        other => descriptor(
            runtime,
            other,
            args,
            requested_outputs,
            metadata,
            fallback_policy,
        ),
    }
}

pub(super) fn try_execute(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    match identity {
        CallableIdentity::Builtin(id) => {
            try_builtin(runtime, id, args, requested_outputs, metadata)
        }
        CallableIdentity::BoundFunction(function) => {
            try_semantic(runtime, function, args, requested_outputs, metadata)
        }
        CallableIdentity::ExternalFunction {
            function,
            display_name,
        } => try_external(
            runtime,
            function,
            display_name,
            args,
            requested_outputs,
            metadata,
        ),
        other => try_descriptor(
            runtime,
            other,
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
    id: BuiltinId,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    Box::pin(invoke_builtin(
        runtime,
        id,
        args,
        requested_outputs,
        metadata.function_input_callsite,
    ))
}

#[inline(never)]
fn try_builtin(
    runtime: Option<&RuntimeContext>,
    id: BuiltinId,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    Box::pin(async move {
        invoke_builtin(
            runtime,
            id,
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
    id: BuiltinId,
    args: Vec<Value>,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<RootedValueSequence, RuntimeError> {
    call_builtin_with_requested_outputs(runtime, &id.0, &args, requested_outputs, input_callsite)
        .await
}

#[inline(never)]
fn semantic(
    runtime: Option<&RuntimeContext>,
    function: FunctionId,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    Box::pin(async move {
        match invoke_semantic(
            runtime,
            function,
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
    function: FunctionId,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    Box::pin(invoke_semantic(
        runtime,
        function,
        args,
        requested_outputs,
        metadata.function_input_callsite,
    ))
}

async fn invoke_semantic(
    runtime: Option<&RuntimeContext>,
    function: FunctionId,
    args: Vec<Value>,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    match crate::callsite::with_function_input_callsite(
        input_callsite,
        crate::user_functions::try_call_semantic_function_in_context(
            runtime,
            function.0,
            &args,
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
fn external(
    runtime: Option<&RuntimeContext>,
    function: FunctionId,
    display_name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, RootedValueSequence> {
    Box::pin(async move {
        match invoke_external(
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
        }
    })
}

#[inline(never)]
fn try_external(
    runtime: Option<&RuntimeContext>,
    function: FunctionId,
    display_name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    Box::pin(invoke_external(
        runtime,
        function,
        display_name,
        args,
        requested_outputs,
        metadata.function_input_callsite,
    ))
}

async fn invoke_external(
    runtime: Option<&RuntimeContext>,
    function: FunctionId,
    display_name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    if let Some(result) = crate::callsite::with_function_input_callsite(
        input_callsite.clone(),
        crate::user_functions::try_call_external_function_in_context(
            runtime,
            crate::user_functions::ExternalFunctionCall {
                function: function.0,
                display_name,
                arguments: args.clone(),
                requested_outputs,
            },
        ),
    )
    .await?
    {
        return result.map(Some);
    }
    match crate::callsite::with_function_input_callsite(
        input_callsite,
        crate::user_functions::try_call_semantic_function_in_context(
            runtime,
            function.0,
            &args,
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
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, RootedValueSequence> {
    Box::pin(invoke_descriptor_required(
        runtime,
        identity,
        args,
        requested_outputs,
        metadata,
        fallback_policy,
    ))
}

#[inline(never)]
fn try_descriptor(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> ExecutionFuture<'_, Option<RootedValueSequence>> {
    Box::pin(invoke_descriptor_optional(
        runtime,
        identity,
        args,
        requested_outputs,
        metadata.function_input_callsite,
        fallback_policy,
    ))
}

async fn invoke_descriptor_required(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    metadata: CallableMetadata,
    fallback_policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    let request = crate::user_functions::CallableRequest::resolved(
        identity.clone(),
        fallback_policy,
        args.clone(),
        requested_outputs,
    )
    .with_function_input_callsite(metadata.function_input_callsite.clone())
    .with_selected_runtime(runtime);
    if let Some(result) = crate::user_functions::try_call_semantic_descriptor(request).await {
        return result;
    }
    let Some(name) = fallback_policy.vm_fallback_name_for(&identity) else {
        return Err(undefined_identity_error(&identity, &metadata));
    };
    forward_named_fallback(
        runtime,
        name,
        args,
        requested_outputs,
        metadata.function_input_callsite,
        metadata.source_ref,
        metadata.source_invocation,
    )
    .await
}

async fn invoke_descriptor_optional(
    runtime: Option<&RuntimeContext>,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let request = crate::user_functions::CallableRequest::resolved(
        identity.clone(),
        fallback_policy,
        args.clone(),
        requested_outputs,
    )
    .with_function_input_callsite(input_callsite.clone())
    .with_selected_runtime(runtime);
    if let Some(result) = crate::user_functions::try_call_semantic_descriptor(request).await {
        return result.map(Some);
    }
    let Some(name) = fallback_policy.vm_fallback_name_for(&identity) else {
        return Ok(None);
    };
    let invoke = crate::callsite::with_callsite(
        input_callsite,
        crate::dispatcher::try_call_builtin_sequence_async_with_outputs(
            &name,
            &args,
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
