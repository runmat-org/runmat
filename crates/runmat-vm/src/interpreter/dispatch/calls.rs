use crate::bytecode::ArgSpec;
use crate::call::builtins as call_builtins;
use crate::call::builtins::ImportedBuiltinResolution;
use crate::call::closures as call_closures;
use crate::call::descriptor::{execute_callable_descriptor, CallableCallKind, CallableDescriptor};
use crate::call::shared::{build_expanded_args_from_specs, expand_brace_values};
use crate::interpreter::debug;
use crate::interpreter::dispatch::exceptions::{redirect_exception_to_catch, ExceptionHandling};
use crate::object::class_def as obj_class_def;
use crate::object::resolve as obj_resolve;
use runmat_builtins::{MException, Value};
use runmat_hir::{CallableFallbackPolicy, CallableIdentity};
use runmat_runtime::RuntimeError;

pub enum BuiltinHandling {
    Completed,
    Caught,
    Uncaught(Box<RuntimeError>),
}

pub enum MethodHandling {
    Completed,
}

pub enum UserCallHandling {
    Completed,
    Caught,
    Uncaught(Box<RuntimeError>),
}

pub(crate) fn normalize_requested_outputs(value: Value, requested_outputs: usize) -> Value {
    // Preserve values for non-singleton requests (including zero). Statement-level
    // display/public-result policy is decided later by core/session plumbing.
    if requested_outputs != 1 {
        return value;
    }
    match value {
        Value::OutputList(mut values) if values.len() == 1 => values.remove(0),
        other => other,
    }
}

pub struct ExceptionRouteContext<'a> {
    pub try_stack: &'a mut Vec<(usize, Option<usize>)>,
    pub vars: &'a mut Vec<Value>,
    pub last_exception: &'a mut Option<MException>,
    pub pc: &'a mut usize,
}

pub struct BuiltinCallContext<'a> {
    pub stack: &'a mut Vec<Value>,
    pub name: &'a str,
    pub arg_count: usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub call_arg_spans: Option<Vec<runmat_hir::Span>>,
    pub imports: &'a [(Vec<String>, bool)],
    pub call_counts: &'a [(usize, usize)],
    pub exception: ExceptionRouteContext<'a>,
}

pub struct UserCallContext<'a> {
    pub stack: &'a mut Vec<Value>,
    pub identity: CallableIdentity,
    pub fallback_policy: CallableFallbackPolicy,
    pub out_count: usize,
    pub exception: ExceptionRouteContext<'a>,
}

pub async fn build_builtin_expand_multi_args(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
) -> Result<Vec<Value>, RuntimeError> {
    build_expanded_args_from_specs(
        stack,
        specs,
        "CallBuiltinExpandMulti requires cell or object for expand_all",
        "CallBuiltinExpandMulti requires cell or object cell access",
        |base| async move { expand_brace_values(base, &[], None).await },
        |base, indices| async move { expand_brace_values(base, &indices, None).await },
    )
    .await
}

pub async fn build_feval_expand_multi_args(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
) -> Result<Vec<Value>, RuntimeError> {
    build_expanded_args_from_specs(
        stack,
        specs,
        "CallFevalExpandMulti requires cell or object for expand_all",
        "CallFevalExpandMulti requires cell or object cell access",
        |base| async move { expand_brace_values(base, &[], None).await },
        |base, indices| async move { expand_brace_values(base, &indices, None).await },
    )
    .await
}

pub async fn build_user_function_expand_multi_args(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
) -> Result<Vec<Value>, RuntimeError> {
    build_expanded_args_from_specs(
        stack,
        specs,
        "CallFunctionExpandMultiOutput requires cell or object for expand_all",
        "CallFunctionExpandMultiOutput requires cell or object cell access",
        |base| async move { expand_brace_values(base, &[], None).await },
        |base, indices| async move { expand_brace_values(base, &indices, None).await },
    )
    .await
}

pub fn handle_builtin_outcome(
    result: Result<Value, RuntimeError>,
    imported: ImportedBuiltinResolution,
    output_hint: usize,
    stack: &mut Vec<Value>,
    ctx: ExceptionRouteContext<'_>,
    refresh_vars: impl Fn(&[Value]),
) -> Result<BuiltinHandling, RuntimeError> {
    let ExceptionRouteContext {
        try_stack,
        vars,
        last_exception,
        pc,
    } = ctx;
    match result {
        Ok(result) => {
            stack.push(normalize_requested_outputs(result, output_hint));
            Ok(BuiltinHandling::Completed)
        }
        Err(err) => match imported {
            ImportedBuiltinResolution::Resolved(value) => {
                stack.push(normalize_requested_outputs(value, output_hint));
                Ok(BuiltinHandling::Completed)
            }
            ImportedBuiltinResolution::Ambiguous(message) => Err(message.into()),
            ImportedBuiltinResolution::NotFound => Ok(
                match redirect_exception_to_catch(
                    err,
                    try_stack,
                    vars,
                    last_exception,
                    pc,
                    refresh_vars,
                ) {
                    ExceptionHandling::Caught => BuiltinHandling::Caught,
                    ExceptionHandling::Uncaught(err) => BuiltinHandling::Uncaught(err),
                },
            ),
        },
    }
}

pub async fn handle_builtin_call_multi(
    ctx: BuiltinCallContext<'_>,
    out_count: usize,
    refresh_vars: impl Fn(&[Value]),
) -> Result<BuiltinHandling, RuntimeError> {
    handle_builtin_call_inner(ctx, refresh_vars, out_count).await
}

async fn handle_builtin_call_inner(
    ctx: BuiltinCallContext<'_>,
    refresh_vars: impl Fn(&[Value]),
    requested_outputs: usize,
) -> Result<BuiltinHandling, RuntimeError> {
    let BuiltinCallContext {
        stack,
        name,
        arg_count,
        source_id,
        call_arg_spans,
        imports,
        call_counts,
        exception,
    } = ctx;
    let ExceptionRouteContext {
        try_stack: _,
        vars: _,
        last_exception,
        pc,
    } = &exception;
    debug::trace_call_builtin(**pc, name, arg_count, stack);
    if let Some(value) = call_builtins::vm_intrinsic_counter_builtin(name, arg_count, call_counts)?
    {
        stack.push(value);
        return Ok(BuiltinHandling::Completed);
    }
    let args = call_builtins::collect_call_args(stack, arg_count)?;

    let _callsite_guard = runmat_runtime::callsite::push_callsite(source_id, call_arg_spans);
    let _output_guard = runmat_runtime::output_context::push_output_count(requested_outputs);

    let prepared_primary = call_builtins::prepare_builtin_args(name, &args).await?;
    let result =
        runmat_runtime::call_builtin_async_with_outputs(name, &prepared_primary, requested_outputs)
            .await;
    let imported = call_builtins::resolve_imported_builtin(
        name,
        imports,
        &prepared_primary,
        requested_outputs,
    )
    .await?;
    if result.is_err() {
        if let Some(err) = call_builtins::rethrow_without_explicit_exception(
            name,
            &args,
            last_exception.as_ref().map(|e| e.identifier.as_str()),
            last_exception.as_ref().map(|e| e.message.as_str()),
        ) {
            return Err(err);
        }
    }
    handle_builtin_outcome(
        result,
        imported,
        requested_outputs,
        stack,
        exception,
        refresh_vars,
    )
}

pub async fn handle_prepared_user_function_call(
    ctx: UserCallContext<'_>,
    args: Vec<Value>,
    refresh_vars: impl Fn(&[Value]),
) -> Result<UserCallHandling, RuntimeError> {
    let UserCallContext {
        stack,
        identity,
        fallback_policy,
        out_count,
        exception,
    } = ctx;
    let ExceptionRouteContext {
        try_stack,
        vars,
        last_exception,
        pc,
    } = exception;
    let descriptor = CallableDescriptor::resolved(
        identity,
        args,
        out_count,
        fallback_policy,
        CallableCallKind::Direct,
    );
    match execute_callable_descriptor(descriptor).await {
        Ok(result) => {
            stack.push(normalize_requested_outputs(result, out_count));
            Ok(UserCallHandling::Completed)
        }
        Err(err) => Ok(
            match redirect_exception_to_catch(
                err,
                try_stack,
                vars,
                last_exception,
                pc,
                refresh_vars,
            ) {
                ExceptionHandling::Caught => UserCallHandling::Caught,
                ExceptionHandling::Uncaught(err) => UserCallHandling::Uncaught(err),
            },
        ),
    }
}

pub async fn handle_user_function_call(
    ctx: UserCallContext<'_>,
    arg_count: usize,
    refresh_vars: impl Fn(&[Value]),
) -> Result<UserCallHandling, RuntimeError> {
    let args = crate::call::builtins::collect_call_args(ctx.stack, arg_count)?;
    handle_prepared_user_function_call(ctx, args, refresh_vars).await
}

pub async fn handle_method_or_member_index_multi_call(
    stack: &mut Vec<Value>,
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    arg_count: usize,
    out_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    handle_method_or_member_index_call_inner(stack, identity, fallback_policy, arg_count, out_count)
        .await
}

async fn handle_method_or_member_index_call_inner(
    stack: &mut Vec<Value>,
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    arg_count: usize,
    requested_outputs: usize,
) -> Result<MethodHandling, RuntimeError> {
    let (base, args) = call_closures::collect_method_args(stack, arg_count)?;
    let _output_guard = runmat_runtime::output_context::push_output_count(requested_outputs);
    let value = call_closures::call_method_or_member_index_with_outputs(
        base,
        identity,
        args,
        requested_outputs,
        fallback_policy,
    )
    .await?;
    stack.push(normalize_requested_outputs(value, requested_outputs));
    Ok(MethodHandling::Completed)
}

pub async fn handle_method_or_member_index_expand_multi_call(
    stack: &mut Vec<Value>,
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    specs: &[ArgSpec],
    requested_outputs: usize,
) -> Result<MethodHandling, RuntimeError> {
    let mut args = build_user_function_expand_multi_args(stack, specs).await?;
    if args.is_empty() {
        return Err(crate::interpreter::errors::mex(
            "MethodCallMissingReceiver",
            "method/member-index call requires a base receiver",
        ));
    }
    let base = args.remove(0);
    let _output_guard = runmat_runtime::output_context::push_output_count(requested_outputs);
    let value = call_closures::call_method_or_member_index_with_outputs(
        base,
        identity,
        args,
        requested_outputs,
        fallback_policy,
    )
    .await?;
    stack.push(normalize_requested_outputs(value, requested_outputs));
    Ok(MethodHandling::Completed)
}

pub fn handle_load_method(
    stack: &mut Vec<Value>,
    name: String,
) -> Result<MethodHandling, RuntimeError> {
    let base = crate::interpreter::stack::pop_value(stack)?;
    let value = call_closures::load_method_closure(base, name)?;
    stack.push(value);
    Ok(MethodHandling::Completed)
}

pub fn handle_create_closure(
    stack: &mut Vec<Value>,
    func_name: String,
    capture_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    call_closures::create_closure(stack, func_name, capture_count)?;
    Ok(MethodHandling::Completed)
}

pub fn handle_create_semantic_closure(
    stack: &mut Vec<Value>,
    function: runmat_hir::FunctionId,
    display_name: String,
    capture_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    call_closures::create_semantic_closure(stack, function, display_name, capture_count)?;
    Ok(MethodHandling::Completed)
}

#[cfg(test)]
mod tests {
    use super::normalize_requested_outputs;
    use runmat_builtins::Value;

    #[test]
    fn normalize_requested_outputs_collapses_singleton_for_single_request() {
        let value = normalize_requested_outputs(Value::OutputList(vec![Value::Num(7.0)]), 1);
        assert_eq!(value, Value::Num(7.0));
    }

    #[test]
    fn normalize_requested_outputs_preserves_value_for_zero_request() {
        let value = normalize_requested_outputs(Value::Num(7.0), 0);
        assert_eq!(value, Value::Num(7.0));
    }
}

pub fn handle_load_static_property(
    stack: &mut Vec<Value>,
    class_name: &str,
    prop: &str,
) -> Result<MethodHandling, RuntimeError> {
    let value = obj_resolve::load_static_member(class_name, prop)?;
    stack.push(value);
    Ok(MethodHandling::Completed)
}

pub fn handle_register_class(
    name: String,
    super_class: Option<String>,
    properties: Vec<(String, bool, String, String)>,
    methods: Vec<(String, String, bool, String)>,
) -> Result<MethodHandling, RuntimeError> {
    obj_class_def::register_class(name, super_class, properties, methods)?;
    Ok(MethodHandling::Completed)
}
