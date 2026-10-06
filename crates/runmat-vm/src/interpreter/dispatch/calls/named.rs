use super::*;

pub(super) async fn execute(
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
    let current_class_context =
        runmat_runtime::class_registry::class_context_for_function(current_function_name);
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
            if runmat_runtime::class_registry::get_class(&class_name).is_some() {
                if let Some((method, owner)) =
                    runmat_runtime::class_registry::lookup_method(&class_name, &method_name)
                {
                    if method.is_static {
                        let allowed = match method.access {
                            MemberAccess::Public => true,
                            MemberAccess::Private => {
                                current_class_context.as_deref() == Some(owner.as_str())
                            }
                            MemberAccess::Protected => {
                                current_class_context.as_ref().is_some_and(|caller_class| {
                                    runmat_runtime::class_registry::is_class_or_subclass(
                                        caller_class,
                                        &owner,
                                    )
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
        if let Some((method, owner)) =
            runmat_runtime::class_registry::lookup_method(class_name, method_name)
        {
            if method.is_static {
                let allowed = match method.access {
                    MemberAccess::Public => true,
                    MemberAccess::Private => {
                        current_class_context.as_deref() == Some(owner.as_str())
                    }
                    MemberAccess::Protected => {
                        current_class_context.as_ref().is_some_and(|caller_class| {
                            runmat_runtime::class_registry::is_class_or_subclass(
                                caller_class,
                                &owner,
                            )
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
            if let Some((method, owner)) =
                runmat_runtime::class_registry::lookup_method(&owner_class, method_name)
            {
                if method.is_static {
                    let allowed = match method.access {
                        MemberAccess::Public => true,
                        MemberAccess::Private => {
                            current_class_context.as_deref() == Some(owner.as_str())
                        }
                        MemberAccess::Protected => {
                            current_class_context.as_ref().is_some_and(|caller_class| {
                                runmat_runtime::class_registry::is_class_or_subclass(
                                    caller_class,
                                    &owner,
                                )
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
    super::resolved::execute(
        stack,
        descriptor,
        out_count,
        ExceptionRouteContext {
            try_stack,
            vars,
            last_exception,
            pc,
        },
        current_class_context,
        refresh_vars,
    )
    .await
}
