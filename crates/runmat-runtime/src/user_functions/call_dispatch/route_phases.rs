use super::*;

#[inline(never)]
pub(super) async fn try_bound_identity(
    function: runmat_types::FunctionId,
    bound: Option<runmat_value::BoundFunctionBinding>,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    function_input_callsite: Option<crate::callsite::CallsiteInfo>,
    selected_runtime: Option<&crate::context::RuntimeContext>,
) -> super::super::CallableAttempt {
    if let Some(binding) = bound {
        let available = matches!(
            &binding.authority,
            runmat_value::BoundFunctionAuthority::SelectedNative { .. }
        ) || current_bound_function_invoker().is_some();
        if !available {
            return std::ops::ControlFlow::Continue(args);
        }
        let selected = async {
            match selected_runtime {
                Some(context) => {
                    try_call_rooted_bound_function_with_environment_in_context(
                        context,
                        binding,
                        args,
                        requested_outputs,
                        None,
                    )
                    .await
                }
                None => try_call_rooted_bound_function(binding, args, requested_outputs).await,
            }
        };
        let result = flatten_callsite_result(
            crate::callsite::with_function_input_callsite(function_input_callsite, selected).await,
        )
        .unwrap_or_else(|| {
            Err(crate::runtime_error::semantic_error(
                "UnavailableBoundFunction",
                "selected bound function has no rooted invoker",
            ))
        });
        return std::ops::ControlFlow::Break(result);
    }
    if current_semantic_function_invoker().is_none() {
        return std::ops::ControlFlow::Continue(args);
    }
    let result = crate::callsite::with_function_input_callsite(
        function_input_callsite,
        try_call_rooted_semantic_function_in_context(
            selected_runtime,
            function.0,
            args,
            requested_outputs,
        ),
    )
    .await
    .unwrap_or_else(|error| Some(Err(error)))
    .unwrap_or_else(|| {
        Err(crate::runtime_error::semantic_error(
            "UnavailableFunction",
            "selected semantic function has no rooted invoker",
        ))
    });
    std::ops::ControlFlow::Break(result)
}

/// Qualified native calls retain the clib → Python order. Their
/// rooted argument owner is transferred once at each Continue boundary.
#[inline(never)]
pub(super) async fn try_external_foreign_adapters(
    name: &str,
    mut args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> super::super::CallableAttempt {
    let Some(context) = crate::context::legacy::active() else {
        return std::ops::ControlFlow::Continue(args);
    };
    args = match crate::foreign::try_invoke_clib(context.clone(), name, args, requested_outputs)
        .await
    {
        crate::foreign::ForeignAttempt::Break(result) => {
            return std::ops::ControlFlow::Break(result)
        }
        crate::foreign::ForeignAttempt::Continue(args) => args,
    };
    match crate::foreign::try_invoke_python(context, name, args, requested_outputs).await {
        crate::foreign::ForeignAttempt::Break(result) => std::ops::ControlFlow::Break(result),
        crate::foreign::ForeignAttempt::Continue(args) => std::ops::ControlFlow::Continue(args),
    }
}

#[inline(never)]
pub(super) async fn try_registered_external_static(
    qualified: &runmat_types::QualifiedName,
    name: &str,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    function_input_callsite: &mut Option<crate::callsite::CallsiteInfo>,
    selected_runtime: Option<&crate::context::RuntimeContext>,
) -> super::super::CallableAttempt {
    // A registered public static method is selected by typed class/member
    // authority before any external adapter or generic name fallback.
    let registered = qualified.0.split_last().and_then(|(member, owner)| {
        if owner.is_empty() || member.0.is_empty() {
            return None;
        }
        let class = runmat_types::ClassIdentity::from_qualified_name(&runmat_types::QualifiedName(
            owner.to_vec(),
        ))
        .ok()?;
        crate::class_registry::registered_public_static_member(
            class,
            runmat_types::MethodName::from(member.0.as_str()),
        )
    });
    if let Some(resolved) = registered.and_then(|registration| {
        crate::dispatcher::observe_registered_static_member(name, registration)
    }) {
        let arguments = crate::call::BuiltinArguments::new(args);
        let result = crate::callsite::with_function_input_callsite(
            function_input_callsite.take(),
            crate::dispatcher::invoke_registered_static_method(
                name,
                resolved,
                &arguments,
                Some(requested_outputs),
                selected_runtime,
            ),
        )
        .await
        .and_then(std::convert::identity);
        std::ops::ControlFlow::Break(result)
    } else {
        std::ops::ControlFlow::Continue(args)
    }
}

#[inline(never)]
pub(super) async fn try_registered_constructor_source(
    class: runmat_types::ClassIdentity,
    name: String,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    function_input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> super::super::CallableAttempt {
    // Only an unresolved external constructor declaration can select class
    // source. A registered constructor never falls through to generic MEX.
    let constructor = class
        .qualified_name()
        .0
        .last()
        .map(|name| runmat_types::MethodName::from(name.0.as_str()))
        .and_then(|method| crate::class_registry::lookup_bound_method(&class, &method))
        .or_else(|| {
            crate::class_registry::lookup_bound_method(
                &class,
                &runmat_types::MethodName::from(class.display_name()),
            )
        });
    if let Some(constructor) = constructor.filter(|method| {
        method.declaring_class == class
            && !method.declaration.is_static
            && matches!(&method.callable, CallableIdentity::ExternalName(_))
            && method.fallback == CallableFallbackPolicy::ExternalBoundary
    }) {
        let declared = &constructor.declaration.function_name;
        let owner_qualified = format!(
            "{}.{}",
            constructor.declaring_class,
            class
                .qualified_name()
                .0
                .last()
                .expect("registered class identity has a final segment")
                .0
        );
        if resolve_semantic_function_by_name(declared).is_none()
            && resolve_semantic_function_by_name(&owner_qualified).is_none()
        {
            return try_load_and_call_dynamic_function(
                name,
                args,
                requested_outputs,
                DynamicFunctionLoadPhase::RegisteredConstructorSource(class),
                function_input_callsite,
            )
            .await;
        }
    }
    std::ops::ControlFlow::Continue(args)
}

#[inline(never)]
pub(super) async fn try_named_fallback(
    identity: &CallableIdentity,
    name: String,
    mut args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    mut function_input_callsite: Option<crate::callsite::CallsiteInfo>,
    selected_runtime: Option<&crate::context::RuntimeContext>,
) -> super::super::CallableAttempt {
    if matches!(
        identity,
        CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::ExternalName(_)
    ) {
        // An already selected private source takes priority; public names
        // keep the loader's MEX-first phase before generic semantic lookup.
        if resolve_semantic_function_by_name(&name)
            .and_then(|function| selected_scoped_name_target(&name, function))
            .is_some()
        {
            args = match try_call_rooted_semantic_name_with_callsite_in_context(
                selected_runtime,
                &name,
                args,
                requested_outputs,
                &mut function_input_callsite,
            )
            .await
            {
                std::ops::ControlFlow::Break(result) => {
                    return std::ops::ControlFlow::Break(result)
                }
                std::ops::ControlFlow::Continue(args) => args,
            };
        }
        args = match try_load_and_call_dynamic_function(
            name.clone(),
            args,
            requested_outputs,
            DynamicFunctionLoadPhase::BeforeSemantic,
            function_input_callsite.clone(),
        )
        .await
        {
            std::ops::ControlFlow::Break(result) => return std::ops::ControlFlow::Break(result),
            std::ops::ControlFlow::Continue(args) => args,
        };
    }
    args = match try_call_rooted_semantic_name_with_callsite_in_context(
        selected_runtime,
        &name,
        args,
        requested_outputs,
        &mut function_input_callsite,
    )
    .await
    {
        std::ops::ControlFlow::Break(result) => return std::ops::ControlFlow::Break(result),
        std::ops::ControlFlow::Continue(args) => args,
    };
    if matches!(
        identity,
        CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::ExternalName(_)
    ) {
        args = match try_load_and_call_dynamic_function(
            name.clone(),
            args,
            requested_outputs,
            DynamicFunctionLoadPhase::AfterSemantic,
            function_input_callsite,
        )
        .await
        {
            std::ops::ControlFlow::Break(result) => return std::ops::ControlFlow::Break(result),
            std::ops::ControlFlow::Continue(args) => args,
        };
    }
    if matches!(
        identity,
        CallableIdentity::ExternalName(_) | CallableIdentity::Imported(_)
    ) {
        if let Some(context) = crate::context::legacy::active() {
            args = match crate::foreign::try_invoke_java(context, &name, args, requested_outputs)
                .await
            {
                crate::foreign::ForeignAttempt::Break(result) => {
                    return std::ops::ControlFlow::Break(result)
                }
                crate::foreign::ForeignAttempt::Continue(returned) => returned,
            };
        }
    }
    std::ops::ControlFlow::Continue(args)
}
