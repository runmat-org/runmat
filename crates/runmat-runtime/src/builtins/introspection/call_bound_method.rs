use crate::execution::RootedValue;
use runmat_builtins::{
    CALL_BOUND_METHOD_ERRORS, CALL_METHOD_ERROR_NAME_INVALID, CALL_METHOD_ERROR_RECEIVER_INVALID,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "__runmat_call_bound_method__",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::call_bound_method"
)]
pub async fn call_bound_method_builtin(
    base: Value,
    method: String,
    scope: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<RootedValue> {
    let scope_class = match scope {
        Value::String(text) => {
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
    let _scope_guard = scope_class.and_then(|class_name| {
        runmat_types::ClassIdentity::new(class_name)
            .ok()
            .map(|identity| crate::push_class_access_context(Some(identity)))
    });
    let class_name = crate::object_receiver_class_name(&base).ok_or_else(|| {
        crate::runtime_descriptor_error(
            "__runmat_call_bound_method__",
            &CALL_METHOD_ERROR_RECEIVER_INVALID,
        )
    })?;
    let method = method.trim().to_string();
    if method.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "__runmat_call_bound_method__",
            &CALL_METHOD_ERROR_NAME_INVALID,
        ));
    }
    let requested_outputs = crate::current_requested_outputs();
    let method_identity = runmat_types::MethodName::from(method.as_str());
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
            runmat_types::MemberAccess::Private => caller.as_ref() == Some(&bound.declaring_class),
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
        let mut args = Vec::with_capacity(1 + rest.len());
        args.push(base);
        args.extend(rest);
        let descriptor = crate::call::descriptor::CallableDescriptor::bound(
            binding,
            args,
            requested_outputs,
            crate::call::descriptor::CallableCallKind::Direct,
        );
        let result = crate::call::descriptor::execute_callable_descriptor(descriptor).await?;
        return crate::call::arguments::project_rooted_legacy_builtin_value_abi(result);
    }
    let mut args = Vec::with_capacity(1 + rest.len());
    args.push(base);
    args.extend(rest);
    if let Some((_resolved, owner)) =
        crate::class_registry::lookup_method(&class_name, &method_identity)
    {
        return crate::dispatch_object_external_member(owner, &method, args, requested_outputs)
            .await;
    }
    crate::dispatch_object_external_member(class_name, &method, args, requested_outputs).await
}
