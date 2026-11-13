use crate::execution::RootedValue;
use runmat_builtins::{
    CALL_BOUND_METHOD_ERRORS, CALL_METHOD_ERROR_NAME_INVALID, CALL_METHOD_ERROR_RECEIVER_INVALID,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "__runmat_call_bound_method__",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::call_bound_method"
)]
pub async fn call_bound_method_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    base: Value,
    method: runmat_value::Utf8Text,
    scope: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<RootedValue> {
    let invoke = call_bound_method_retained(runtime.as_ref(), base, method, scope, rest);
    match runtime.as_ref() {
        Some(runtime) => runtime.scope(invoke).await,
        None => invoke.await,
    }
}

async fn call_bound_method_retained(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    method: runmat_value::Utf8Text,
    scope: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<RootedValue> {
    let scope_class = match scope {
        Value::String(runmat_value::StringElement::Text(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        other => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "__runmat_call_bound_method__",
                &CALL_BOUND_METHOD_ERRORS[0],
                format!("expected scope string, got {other:?}"),
            ))
        }
    };
    let _scope_guard = scope_class
        .and_then(|class_name| {
            runmat_types::ClassIdentity::new(class_name)
                .ok()
                .map(|identity| crate::push_class_access_context(Some(identity)))
        })
        .transpose()?;
    let class_name = crate::object_receiver_class_name(&base).ok_or_else(|| {
        crate::runtime_descriptor_error(
            "__runmat_call_bound_method__",
            &CALL_METHOD_ERROR_RECEIVER_INVALID,
        )
    })?;
    let method = method.trim();
    if method.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "__runmat_call_bound_method__",
            &CALL_METHOD_ERROR_NAME_INVALID,
        ));
    }
    let requested_outputs = crate::current_requested_outputs();
    let method_identity = runmat_types::MethodName::from(method);
    if let Some((bound, binding)) =
        crate::class_registry::lookup_bound_method(&class_name, &method_identity).and_then(
            |bound| {
                let binding = bound.bound_function.clone()?;
                Some((bound, binding))
            },
        )
    {
        let caller = crate::class_access_context();
        let accessible = match bound.declaration.access {
            runmat_types::MemberAccess::Public => true,
            runmat_types::MemberAccess::Private => {
                caller.as_deref() == Some(&bound.declaring_class)
            }
            runmat_types::MemberAccess::Protected => caller.as_ref().is_some_and(|caller| {
                crate::class_registry::is_class_or_subclass(caller, &bound.declaring_class)
            }),
        };
        if bound.declaration.is_static {
            return Err(crate::runtime_error::semantic_error(
                "MethodStaticOnInstance",
                "bound static method requires a class reference",
            ));
        }
        if !accessible {
            return Err(crate::runtime_error::semantic_error(
                "MethodPrivate",
                "bound method is inaccessible on this receiver",
            ));
        }
        let args = prepend_receiver(base, &rest, runtime).await?;
        let result = crate::call::descriptor::execute_retained_bound_in_context(
            runtime,
            binding,
            args,
            requested_outputs,
            crate::call::descriptor::CallableCallKind::Direct,
        )
        .await?;
        return crate::call::arguments::project_rooted_legacy_builtin_value_abi(result);
    }
    let args = prepend_receiver(base, &rest, runtime).await?;
    let owner = crate::class_registry::lookup_method(&class_name, &method_identity)
        .map_or(class_name, |(_, owner)| owner);
    crate::dispatch_retained_callable_with_policy_in_context(
        runtime,
        crate::class_member_identity(&owner, method),
        runmat_types::CallableFallbackPolicy::ExternalBoundary,
        args,
        requested_outputs,
    )
    .await
}
async fn prepend_receiver(
    base: Value,
    rest: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<crate::call::BuiltinArguments> {
    let rooted = match runtime {
        Some(runtime) => RootedValue::in_context(base, runtime)?,
        None => {
            use crate::context::host_allocation::{
                host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
                HostAllocationReceipt,
            };
            let error = |source| {
                host_allocation_runtime_error(
                    None,
                    "RunMat:Call:Arguments",
                    "bound receiver policy unavailable",
                    source,
                )
            };
            if HostAllocationReceipt::policy(None).map_err(error)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(error(HostAllocationError::MissingManagedOwner));
            }
            RootedValue::new(base).map_err(|e| e.into_runtime_error())?
        }
    };
    crate::builtins::common::mapped_callable::prepend_rooted_argument(rest, rooted, runtime).await
}
