use crate::bytecode::instr::PropertyDefaultLiteral;
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
use runmat_builtins::{Access, MException, Value};
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

fn current_class_context_from_function_name(current_function_name: &str) -> Option<String> {
    if current_function_name.is_empty() {
        return None;
    }
    if let Some((class_name, method_name)) = current_function_name.rsplit_once('.') {
        if !class_name.is_empty()
            && !method_name.is_empty()
            && runmat_builtins::get_class(class_name).is_some()
        {
            return Some(class_name.to_string());
        }
    }
    runmat_builtins::class_names()
        .into_iter()
        .find(|class_name| {
            runmat_builtins::get_class(class_name).is_some_and(|class_def| {
                class_def.methods.values().any(|method| {
                    method.function_name == current_function_name
                        || method
                            .function_name
                            .strip_prefix(class_name)
                            .is_some_and(|suffix| {
                                suffix
                                    .strip_prefix('.')
                                    .is_some_and(|name| name == current_function_name)
                            })
                        || method
                            .function_name
                            .rsplit_once('.')
                            .is_some_and(|(_, name)| name == current_function_name)
                })
            })
        })
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
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub current_function_name: &'a str,
    pub exception: ExceptionRouteContext<'a>,
}

pub struct UserCallContext<'a> {
    pub stack: &'a mut Vec<Value>,
    pub identity: CallableIdentity,
    pub fallback_policy: CallableFallbackPolicy,
    pub out_count: usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub call_arg_spans: Option<Vec<runmat_hir::Span>>,
    pub current_function_name: &'a str,
    pub imports: &'a [(Vec<String>, bool)],
    pub exception: ExceptionRouteContext<'a>,
}

pub struct WorkspaceFirstCallContext<'a> {
    pub stack: &'a mut Vec<Value>,
    pub workspace_name: &'a str,
    pub identity: CallableIdentity,
    pub fallback_policy: CallableFallbackPolicy,
    pub out_count: usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub call_arg_spans: Option<Vec<runmat_hir::Span>>,
    pub current_function_name: &'a str,
    pub imports: &'a [(Vec<String>, bool)],
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub exception: ExceptionRouteContext<'a>,
}

fn imported_static_method_owner(
    imports: &[(Vec<String>, bool)],
    method_name: &str,
) -> Result<Option<String>, RuntimeError> {
    let mut owners = Vec::new();
    for (path, wildcard) in imports {
        if *wildcard {
            if path.is_empty() {
                continue;
            }
            let class_name = path.join(".");
            if let Some((method, owner)) = runmat_builtins::lookup_method(&class_name, method_name)
            {
                if method.is_static && !owners.iter().any(|existing| existing == &owner) {
                    owners.push(owner);
                }
            }
            continue;
        }
        if path.len() < 2 || path[path.len() - 1] != method_name {
            continue;
        }
        let class_name = path[..path.len() - 1].join(".");
        if let Some((method, owner)) = runmat_builtins::lookup_method(&class_name, method_name) {
            if method.is_static && !owners.iter().any(|existing| existing == &owner) {
                owners.push(owner);
            }
        }
    }
    if owners.len() > 1 {
        return Err(crate::interpreter::errors::mex(
            "AmbiguousImport",
            &format!(
                "ambiguous static method '{}' via imports: {}",
                method_name,
                owners.join(", ")
            ),
        ));
    }
    Ok(owners.into_iter().next())
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
            ImportedBuiltinResolution::Ambiguous(err) => Err(err),
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
        function_registry,
        current_function_name,
        exception,
    } = ctx;
    let ExceptionRouteContext {
        try_stack: _,
        vars: _,
        last_exception,
        pc,
    } = &exception;
    debug::trace_call_builtin(**pc, name, arg_count, stack);
    if call_builtins::is_vm_intrinsic_builtin(name) {
        let result = call_builtins::vm_intrinsic_builtin(stack, name, arg_count, call_counts);
        return handle_builtin_outcome(
            result,
            ImportedBuiltinResolution::NotFound,
            requested_outputs,
            stack,
            exception,
            refresh_vars,
        );
    }
    if call_builtins::is_vm_dynamic_workspace_builtin(name) {
        let result = call_builtins::vm_dynamic_workspace_builtin(
            stack,
            name,
            arg_count,
            requested_outputs,
            function_registry,
            source_id,
        )
        .await;
        return handle_builtin_outcome(
            result,
            ImportedBuiltinResolution::NotFound,
            requested_outputs,
            stack,
            exception,
            refresh_vars,
        );
    }
    let args = call_builtins::collect_call_args(stack, arg_count)?;

    let _callsite_guard = runmat_runtime::callsite::push_callsite(source_id, call_arg_spans);
    let _output_guard = runmat_runtime::output_context::push_output_count(requested_outputs);
    let current_class_context = current_class_context_from_function_name(current_function_name);
    let _access_guard = current_class_context
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)));

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
        source_id,
        call_arg_spans,
        current_function_name,
        imports,
        exception,
    } = ctx;
    let current_class_context = current_class_context_from_function_name(current_function_name);
    let _function_input_callsite_guard =
        runmat_runtime::callsite::push_function_input_callsite(source_id, call_arg_spans);
    let ExceptionRouteContext {
        try_stack,
        vars,
        last_exception,
        pc,
    } = exception;
    let static_candidate = match &identity {
        runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
            if segments.len() >= 2 =>
        {
            let class_name = segments[..segments.len() - 1]
                .iter()
                .map(|segment| segment.0.as_str())
                .collect::<Vec<_>>()
                .join(".");
            Some((class_name, segments[segments.len() - 1].0.clone()))
        }
        runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name))
            if name.contains('.') =>
        {
            name.rsplit_once('.')
                .map(|(class_name, method_name)| (class_name.to_string(), method_name.to_string()))
        }
        _ => None,
    };
    if current_class_context.is_some() {
        if let Some((class_name, method_name)) = static_candidate {
            if runmat_builtins::get_class(&class_name).is_some() {
                if let Some((method, owner)) =
                    runmat_builtins::lookup_method(&class_name, &method_name)
                {
                    if method.is_static {
                        let allowed = match method.access {
                            Access::Public => true,
                            Access::Private => {
                                current_class_context.as_deref() == Some(owner.as_str())
                            }
                            Access::Protected => {
                                current_class_context.as_ref().is_some_and(|caller_class| {
                                    runmat_builtins::is_class_or_subclass(caller_class, &owner)
                                })
                            }
                        };
                        if !allowed {
                            return Err(crate::interpreter::errors::mex(
                                "MethodPrivate",
                                &format!("Method '{}' is private", method_name),
                            ));
                        }
                        let method_identity = if method.function_name.contains('.') {
                            runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(
                                method
                                    .function_name
                                    .split('.')
                                    .map(|segment| {
                                        runmat_hir::SymbolName(segment.trim().to_string())
                                    })
                                    .collect(),
                            ))
                        } else {
                            runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(
                                method.function_name.clone(),
                            ))
                        };
                        let static_descriptor = CallableDescriptor::resolved(
                            method_identity,
                            args,
                            out_count,
                            runmat_hir::CallableFallbackPolicy::ExternalBoundary,
                            CallableCallKind::Direct,
                        );
                        let result = execute_callable_descriptor(static_descriptor).await?;
                        stack.push(normalize_requested_outputs(result, out_count));
                        return Ok(UserCallHandling::Completed);
                    }
                }
            }
        }
    }

    let local_method_candidate = match &identity {
        runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name))
            if !name.contains('.') && !name.trim().is_empty() =>
        {
            Some(name.trim().to_string())
        }
        runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
            if segments.len() == 1 && !segments[0].0.trim().is_empty() =>
        {
            Some(segments[0].0.trim().to_string())
        }
        _ => None,
    };
    if let (Some(class_name), Some(method_name)) = (
        current_class_context.as_ref(),
        local_method_candidate.as_ref(),
    ) {
        if let Some((method, owner)) = runmat_builtins::lookup_method(class_name, method_name) {
            if method.is_static {
                let allowed = match method.access {
                    Access::Public => true,
                    Access::Private => current_class_context.as_deref() == Some(owner.as_str()),
                    Access::Protected => {
                        current_class_context.as_ref().is_some_and(|caller_class| {
                            runmat_builtins::is_class_or_subclass(caller_class, &owner)
                        })
                    }
                };
                if !allowed {
                    return Err(crate::interpreter::errors::mex(
                        "MethodPrivate",
                        &format!("Method '{}' is private", method_name),
                    ));
                }
                let method_identity = if method.function_name.contains('.') {
                    runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(
                        method
                            .function_name
                            .split('.')
                            .map(|segment| runmat_hir::SymbolName(segment.trim().to_string()))
                            .collect(),
                    ))
                } else {
                    runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(
                        method.function_name.clone(),
                    ))
                };
                let static_descriptor = CallableDescriptor::resolved(
                    method_identity,
                    args,
                    out_count,
                    runmat_hir::CallableFallbackPolicy::ExternalBoundary,
                    CallableCallKind::Direct,
                );
                let result = execute_callable_descriptor(static_descriptor).await?;
                stack.push(normalize_requested_outputs(result, out_count));
                return Ok(UserCallHandling::Completed);
            }
        }
    }
    if let Some(method_name) = local_method_candidate.as_ref() {
        if let Some(owner_class) = imported_static_method_owner(imports, method_name)? {
            if let Some((method, owner)) = runmat_builtins::lookup_method(&owner_class, method_name)
            {
                if method.is_static {
                    let allowed = match method.access {
                        Access::Public => true,
                        Access::Private => current_class_context.as_deref() == Some(owner.as_str()),
                        Access::Protected => {
                            current_class_context.as_ref().is_some_and(|caller_class| {
                                runmat_builtins::is_class_or_subclass(caller_class, &owner)
                            })
                        }
                    };
                    if !allowed {
                        return Err(crate::interpreter::errors::mex(
                            "MethodPrivate",
                            &format!("Method '{}' is private", method_name),
                        ));
                    }
                    let method_identity = if method.function_name.contains('.') {
                        runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(
                            method
                                .function_name
                                .split('.')
                                .map(|segment| runmat_hir::SymbolName(segment.trim().to_string()))
                                .collect(),
                        ))
                    } else {
                        runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(
                            method.function_name.clone(),
                        ))
                    };
                    let static_descriptor = CallableDescriptor::resolved(
                        method_identity,
                        args,
                        out_count,
                        runmat_hir::CallableFallbackPolicy::ExternalBoundary,
                        CallableCallKind::Direct,
                    );
                    let result = execute_callable_descriptor(static_descriptor).await?;
                    stack.push(normalize_requested_outputs(result, out_count));
                    return Ok(UserCallHandling::Completed);
                }
            }
        }
    }

    let descriptor = CallableDescriptor::resolved(
        identity,
        args,
        out_count,
        fallback_policy,
        CallableCallKind::Direct,
    );
    let _access_guard = current_class_context
        .clone()
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)));
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

pub async fn handle_workspace_first_prepared_call(
    ctx: WorkspaceFirstCallContext<'_>,
    args: Vec<Value>,
    refresh_vars: impl Fn(&[Value]),
) -> Result<UserCallHandling, RuntimeError> {
    let WorkspaceFirstCallContext {
        stack,
        workspace_name,
        identity,
        fallback_policy,
        out_count,
        source_id,
        call_arg_spans,
        current_function_name,
        imports,
        function_registry,
        exception,
    } = ctx;

    if let Some(base) = crate::runtime::workspace::workspace_lookup(workspace_name) {
        let _callsite_guard = runmat_runtime::callsite::push_callsite(source_id, call_arg_spans);
        let _output_guard = runmat_runtime::output_context::push_output_count(out_count);
        let result =
            super::indexing::paren_index_value(base, args, out_count, function_registry).await;
        let ExceptionRouteContext {
            try_stack,
            vars,
            last_exception,
            pc,
        } = exception;
        return match result {
            Ok(value) => {
                stack.push(normalize_requested_outputs(value, out_count));
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
        };
    }

    if let CallableIdentity::Builtin(id) = &identity {
        if call_builtins::is_vm_dynamic_workspace_builtin(&id.0) {
            let mut temp_stack = args;
            let arg_count = temp_stack.len();
            let result = call_builtins::vm_dynamic_workspace_builtin(
                &mut temp_stack,
                &id.0,
                arg_count,
                out_count,
                function_registry,
                source_id,
            )
            .await;
            let ExceptionRouteContext {
                try_stack,
                vars,
                last_exception,
                pc,
            } = exception;
            return match result {
                Ok(value) => {
                    stack.push(normalize_requested_outputs(value, out_count));
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
            };
        }
    }

    handle_prepared_user_function_call(
        UserCallContext {
            stack,
            identity,
            fallback_policy,
            out_count,
            source_id,
            call_arg_spans,
            current_function_name,
            imports,
            exception,
        },
        args,
        refresh_vars,
    )
    .await
}

pub async fn handle_method_or_member_index_multi_call(
    stack: &mut Vec<Value>,
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    arg_count: usize,
    out_count: usize,
    current_function_name: &str,
) -> Result<MethodHandling, RuntimeError> {
    handle_method_or_member_index_call_inner(
        stack,
        identity,
        fallback_policy,
        arg_count,
        out_count,
        current_function_name,
    )
    .await
}

async fn handle_method_or_member_index_call_inner(
    stack: &mut Vec<Value>,
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    arg_count: usize,
    requested_outputs: usize,
    current_function_name: &str,
) -> Result<MethodHandling, RuntimeError> {
    let (base, args) = call_closures::collect_method_args(stack, arg_count)?;
    let _output_guard = runmat_runtime::output_context::push_output_count(requested_outputs);
    let current_class_context = current_class_context_from_function_name(current_function_name);
    let _access_guard = current_class_context
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)));
    let value = call_closures::call_method_or_member_index_with_outputs(
        base,
        identity,
        args,
        requested_outputs,
        (!current_function_name.is_empty()).then_some(current_function_name),
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
    current_function_name: &str,
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
    let current_class_context = current_class_context_from_function_name(current_function_name);
    let _access_guard = current_class_context
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)));
    let value = call_closures::call_method_or_member_index_with_outputs(
        base,
        identity,
        args,
        requested_outputs,
        (!current_function_name.is_empty()).then_some(current_function_name),
        fallback_policy,
    )
    .await?;
    stack.push(normalize_requested_outputs(value, requested_outputs));
    Ok(MethodHandling::Completed)
}

pub fn handle_load_method(
    stack: &mut Vec<Value>,
    name: String,
    current_function_name: &str,
) -> Result<MethodHandling, RuntimeError> {
    let base = crate::interpreter::stack::pop_value(stack)?;
    let value = call_closures::load_method_closure(
        base,
        name,
        (!current_function_name.is_empty()).then_some(current_function_name),
    )?;
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

pub fn handle_load_static_property(
    stack: &mut Vec<Value>,
    class_name: &str,
    prop: &str,
) -> Result<MethodHandling, RuntimeError> {
    let value = obj_resolve::load_static_member(class_name, prop, None)?;
    stack.push(value);
    Ok(MethodHandling::Completed)
}

pub fn handle_register_class(
    name: String,
    super_class: Option<String>,
    is_sealed: bool,
    is_abstract: bool,
    properties: Vec<(
        String,
        bool,
        bool,
        Option<PropertyDefaultLiteral>,
        String,
        String,
    )>,
    methods: Vec<(String, String, bool, bool, bool, String)>,
    enumerations: Vec<String>,
) -> Result<MethodHandling, RuntimeError> {
    let properties = properties
        .into_iter()
        .map(
            |(name, is_static, is_constant, default_literal, get_access, set_access)| {
                let default_value = default_literal.map(|literal| match literal {
                    PropertyDefaultLiteral::Num(value) => Value::Num(value),
                    PropertyDefaultLiteral::Bool(value) => Value::Bool(value),
                    PropertyDefaultLiteral::String(value) => Value::String(value),
                });
                (
                    name,
                    is_static,
                    is_constant,
                    default_value,
                    get_access,
                    set_access,
                )
            },
        )
        .collect();
    obj_class_def::register_class(
        name,
        super_class,
        is_sealed,
        is_abstract,
        properties,
        methods,
        enumerations,
    )?;
    Ok(MethodHandling::Completed)
}

#[cfg(test)]
mod tests {
    use super::{handle_builtin_outcome, normalize_requested_outputs, ExceptionRouteContext};
    use crate::call::builtins::ImportedBuiltinResolution;
    use crate::interpreter::errors::mex;
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

    #[test]
    fn handle_builtin_outcome_preserves_ambiguous_import_identifier() {
        let mut try_stack: Vec<(usize, Option<usize>)> = Vec::new();
        let mut vars: Vec<Value> = Vec::new();
        let mut last_exception = None;
        let mut pc = 0usize;
        let mut stack: Vec<Value> = Vec::new();

        let outcome = handle_builtin_outcome(
            Err("primary builtin failure".into()),
            ImportedBuiltinResolution::Ambiguous(mex(
                "AmbiguousBuiltinImport",
                "ambiguous builtin via imports",
            )),
            1,
            &mut stack,
            ExceptionRouteContext {
                try_stack: &mut try_stack,
                vars: &mut vars,
                last_exception: &mut last_exception,
                pc: &mut pc,
            },
            |_| {},
        );
        let err = match outcome {
            Err(err) => err,
            Ok(_) => panic!("ambiguous imported builtin must surface explicit identifier"),
        };
        assert_eq!(err.identifier(), Some("RunMat:AmbiguousBuiltinImport"));
        assert!(err.message().contains("ambiguous builtin via imports"));
    }
}
