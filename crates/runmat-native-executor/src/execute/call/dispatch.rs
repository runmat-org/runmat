use super::*;
use runmat_mir::MirCallee;
use runmat_runtime::call::descriptor::{CallableCallKind, CallableDescriptor};

// Each callee kind builds only its own selected future, after argument
// materialization has unwound. The scheduler still polls the future only
// after generated native code returns its suspension.
#[inline(never)]
pub(super) fn prepared(
    state: &mut HostState,
    call: &MirCall,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
    arguments: Vec<Value>,
    input_callsite: CallsiteInfo,
) -> NativeExecutorResult<OutputListStorage> {
    if matches!(
        call.syntax,
        runmat_types::CallSyntax::Method | runmat_types::CallSyntax::DottedInvoke
    ) {
        if let MirCallee::Static(identity) = &call.callee {
            if !matches!(
                identity,
                runmat_types::CallableIdentity::BoundFunction(_)
                    | runmat_types::CallableIdentity::ExternalFunction { .. }
            ) {
                return dispatch_method(
                    state,
                    identity,
                    arguments,
                    requested_outputs,
                    embedded,
                    call.fallback_policy,
                );
            }
        }
    }
    match &call.callee {
        MirCallee::Static(identity) => dispatch_static(
            state,
            identity,
            arguments,
            requested_outputs,
            embedded,
            input_callsite,
            call.fallback_policy,
        ),
        MirCallee::Dynamic(operand) => dispatch_dynamic(
            state,
            operand,
            arguments,
            requested_outputs,
            embedded,
            input_callsite,
        ),
        MirCallee::SuperConstructor {
            current_class,
            super_class,
        } => dispatch_super_constructor(
            state,
            current_class,
            super_class,
            arguments,
            requested_outputs,
            embedded,
        ),
        MirCallee::SuperMethod {
            current_class,
            super_class,
            method,
        } => dispatch_super_method(
            state,
            current_class,
            super_class,
            method,
            arguments,
            requested_outputs,
            embedded,
        ),
    }
}

fn local_program_function(
    identity: &runmat_types::CallableIdentity,
) -> NativeExecutorResult<Option<runmat_types::FunctionId>> {
    let function = match identity {
        runmat_types::CallableIdentity::BoundFunction(function)
        | runmat_types::CallableIdentity::AnonymousFunction(function) => function.0,
        _ => return Ok(None),
    };
    Ok(Some(runmat_types::FunctionId(function)))
}

#[inline(never)]
fn dispatch_method(
    state: &mut HostState,
    identity: &runmat_types::CallableIdentity,
    mut arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
    fallback_policy: runmat_types::CallableFallbackPolicy,
) -> NativeExecutorResult<OutputListStorage> {
    if arguments.is_empty() {
        return Err(NativeExecutorError::Host(
            "method/member-index call requires a base receiver".into(),
        ));
    }
    let base = arguments.remove(0);
    let caller = state.function.name.clone();
    let class_context = runmat_runtime::class_registry::class_context_for_function(&caller);
    let runtime = state.runtime.clone();
    complete_rooted_call(
        state,
        embedded.cloned(),
        {
            let identity = identity.clone();
            with_output_count(
                requested_outputs,
                with_class_access_context(class_context, async move {
                    runmat_runtime::object::dispatch::call_method_or_member_index_with_outputs_in_context(
                        &runtime,
                        base,
                        identity,
                        arguments,
                        requested_outputs,
                        (!caller.is_empty()).then_some(caller.as_str()),
                        fallback_policy,
                    )
                    .await
                }),
            )
        },
        requested_outputs,
    )
}

#[inline(never)]
fn dispatch_static(
    state: &mut HostState,
    identity: &runmat_types::CallableIdentity,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
    input_callsite: CallsiteInfo,
    fallback_policy: runmat_types::CallableFallbackPolicy,
) -> NativeExecutorResult<OutputListStorage> {
    if let Some(function) = local_program_function(identity)? {
        if let Some(captures) = state.lexical_captures(function)? {
            return dispatch_lexical(
                state,
                function,
                captures,
                arguments,
                requested_outputs,
                embedded,
                input_callsite,
            );
        }
    }
    dispatch_descriptor(
        state,
        identity,
        arguments,
        requested_outputs,
        embedded,
        input_callsite,
        fallback_policy,
    )
}

#[inline(never)]
fn dispatch_lexical(
    state: &mut HostState,
    function: runmat_types::FunctionId,
    captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
    input_callsite: CallsiteInfo,
) -> NativeExecutorResult<OutputListStorage> {
    let invoker =
        runmat_runtime::user_functions::current_lexical_function_invoker().ok_or_else(|| {
            NativeExecutorError::Host("native nested call has no lexical function invoker".into())
        })?;
    let runtime = state.runtime.clone();
    let lexical_call = runmat_runtime::call::lexical::LexicalCall {
        function: function.0,
        captures,
        environment: state.lexical_environment.clone(),
        arguments,
        requested_outputs,
    };
    let _call_roots = runmat_runtime::execution::RootedValueLease::for_traced(&lexical_call)
        .map_err(|error| error.into_runtime_error())?;
    let selected_runtime = runtime.clone();
    let future =
        runmat_runtime::execution::admit_rooted_result_with(&runtime, move || async move {
            let _call_roots = _call_roots;
            runmat_runtime::callsite::with_function_input_callsite(
                Some(input_callsite),
                async move { invoker(Some(selected_runtime), lexical_call)?.await },
            )
            .await?
        })?;
    super::super::call_suspension::begin_lexical_call(
        state,
        embedded.cloned(),
        future,
        requested_outputs,
    )
    .map(OutputListStorage::unbounded)
}

#[inline(never)]
fn dispatch_descriptor(
    state: &mut HostState,
    identity: &runmat_types::CallableIdentity,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
    input_callsite: CallsiteInfo,
    fallback_policy: runmat_types::CallableFallbackPolicy,
) -> NativeExecutorResult<OutputListStorage> {
    let mut descriptor = CallableDescriptor::resolved(
        identity.clone(),
        arguments,
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    )
    .with_function_input_callsite(input_callsite);
    if let Some(selected) = state
        .runtime
        .selected_graph_function_binding(state.function.id, identity)
    {
        descriptor.target =
            runmat_runtime::call::descriptor::CallableTarget::Bound(selected.binding);
    }
    let runtime = state.runtime.clone();
    complete_rooted_call(
        state,
        embedded.cloned(),
        async move {
            runmat_runtime::call::descriptor::execute_callable_descriptor_in_context(
                &runtime, descriptor,
            )
            .await
        },
        requested_outputs,
    )
}

#[inline(never)]
fn dispatch_dynamic(
    state: &mut HostState,
    operand: &runmat_mir::MirOperand,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
    input_callsite: CallsiteInfo,
) -> NativeExecutorResult<OutputListStorage> {
    let target = materialize_operand(state, operand)?;
    let runtime = state.runtime.clone();
    complete_rooted_call(
        state,
        embedded.cloned(),
        async move {
            runmat_runtime::call_feval_sequence_async_with_outputs_and_callsite_in_context(
                &runtime,
                target,
                &arguments,
                requested_outputs,
                Some(input_callsite),
            )
            .await
        },
        requested_outputs,
    )
}

#[inline(never)]
fn dispatch_super_constructor(
    state: &mut HostState,
    current_class: &runmat_types::ClassIdentity,
    super_class: &runmat_types::ClassIdentity,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<OutputListStorage> {
    let current_class = current_class.clone();
    let super_class = super_class.clone();
    let runtime = state.runtime.clone();
    complete_rooted_call(
        state,
        embedded.cloned(),
        with_output_count(requested_outputs, async move {
            runmat_runtime::call_super_constructor(&runtime, current_class, super_class, arguments)
                .await?
                .try_map_owned(
                    runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                    |error| error.into_runtime_error(),
                )
        }),
        requested_outputs,
    )
}

#[inline(never)]
fn dispatch_super_method(
    state: &mut HostState,
    current_class: &runmat_types::ClassIdentity,
    super_class: &runmat_types::ClassIdentity,
    method: &runmat_types::MethodName,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<OutputListStorage> {
    let current_class = current_class.clone();
    let super_class = super_class.clone();
    let method = method.clone();
    let runtime = state.runtime.clone();
    complete_rooted_call(
        state,
        embedded.cloned(),
        with_output_count(requested_outputs, async move {
            runmat_runtime::call_super_method_in_context(
                &runtime,
                current_class,
                super_class,
                method,
                arguments,
            )
            .await?
            .try_map_owned(
                runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                |error| error.into_runtime_error(),
            )
        }),
        requested_outputs,
    )
}
