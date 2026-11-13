use crate::call::closures::method_access_permitted;
use crate::call::identity::external_qualified_identity;
use crate::execution::{RootedValue, RootedValueSequence};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::CallableFallbackPolicy;
use runmat_value::Value;

/// Outcome of resolving and invoking an operator overload.
///
/// `None` is produced only when resolution completed without selecting a
/// callee. Once a callee is selected, its result or error is returned directly;
/// callers must not treat an invocation error as permission to try another
/// implementation.
pub type OperatorMethodDispatch = Option<RootedValue>;

pub(super) fn is_operator_overload_method(name: &runmat_types::MethodName) -> bool {
    runmat_types::OperatorOverloadMethod::from_name(name).is_some()
}

async fn try_operator_method_with_outputs(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: &Value,
    method_args: crate::call::BuiltinArguments,
    name: &str,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let class_name = match receiver {
        Value::Object(object) => object.class_name().clone(),
        Value::HandleObject(handle) => handle.class_name.as_identity().clone(),
        _ => {
            return Err(semantic_error(
                "InvalidObjectDispatch",
                "operator dispatch requires an object operand",
            ));
        }
    };
    let method_name = runmat_types::MethodName::from(name);
    if crate::object::protocol::selected_receiver_has_authority(receiver)? {
        let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
            caller_function_name,
        );
        let Some(method) = crate::object::protocol::resolve_declared_object_method(
            receiver,
            &method_name,
            &access,
        )?
        else {
            return Ok(None);
        };
        let result = crate::object::protocol::invoke_resolved_object_method_retained(
            runtime,
            &method,
            method_args,
            requested_outputs,
        )
        .await?;
        return Ok(Some(result));
    }
    if let Some(bound) = crate::class_registry::lookup_bound_method(&class_name, &method_name) {
        let declaration = &bound.declaration;
        if declaration.is_static {
            return Err(semantic_error(
                "MethodStaticOnInstance",
                format!(
                    "Method '{}' is static; use classref({}).{}",
                    name, class_name, name
                ),
            ));
        }
        if !method_access_permitted(
            &bound.declaring_class,
            &declaration.access,
            caller_function_name,
        ) {
            return Err(semantic_error(
                "MethodPrivate",
                format!("Method '{}' is private", name),
            ));
        }
        return super::call_registered_method(runtime, bound, method_args, requested_outputs)
            .await
            .map(Some);
    }

    super::try_call_identity_with_policy(
        runtime,
        external_qualified_identity(class_name.display_name(), name),
        method_args,
        requested_outputs,
        CallableFallbackPolicy::ExternalBoundary,
    )
    .await
}

fn missing_operator_method(name: &str) -> RuntimeError {
    semantic_error(
        "UndefinedFunction",
        format!("selected class has no admitted '{name}' operator method"),
    )
}

pub async fn call_rhs_operator_method_ordered_with_outputs(
    lhs: Value,
    rhs: Value,
    name: String,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    call_rhs_operator_method_ordered_with_context(
        None,
        lhs,
        rhs,
        name,
        requested_outputs,
        caller_function_name,
    )
    .await
}

pub async fn call_rhs_operator_method_ordered_with_outputs_in_context(
    runtime: &crate::context::RuntimeContext,
    lhs: Value,
    rhs: Value,
    name: String,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    call_rhs_operator_method_ordered_with_context(
        Some(runtime),
        lhs,
        rhs,
        name,
        requested_outputs,
        caller_function_name,
    )
    .await
}

async fn call_rhs_operator_method_ordered_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
    name: String,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    let work = call_rhs_operator_method_ordered_in_scope(
        runtime,
        lhs,
        rhs,
        name,
        requested_outputs,
        caller_function_name,
    );
    match runtime {
        Some(runtime) => runtime.scope(work).await,
        None => work.await,
    }
}

async fn call_rhs_operator_method_ordered_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
    name: String,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    let receiver = rhs.clone();
    try_operator_method_with_outputs(
        runtime,
        &receiver,
        crate::call::BuiltinArguments::from_borrowed(&[lhs, rhs], runtime)?,
        &name,
        requested_outputs,
        caller_function_name,
    )
    .await?
    .ok_or_else(|| missing_operator_method(&name))
}

pub async fn call_object_operator_method(
    base: Value,
    method: &str,
    arg: Value,
) -> Result<RootedValue, RuntimeError> {
    match try_call_object_operator_method(base, method, arg).await? {
        Some(value) => Ok(value),
        None => Err(missing_operator_method(method)),
    }
}

pub async fn call_object_operator_method_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    method: &str,
    arg: Value,
) -> Result<RootedValue, RuntimeError> {
    match try_call_object_operator_method_in_context(runtime, base, method, arg).await? {
        Some(value) => Ok(value),
        None => Err(missing_operator_method(method)),
    }
}

pub async fn try_call_object_operator_method(
    base: Value,
    method: &str,
    arg: Value,
) -> Result<OperatorMethodDispatch, RuntimeError> {
    let receiver = base.clone();
    operator_method_dispatch(
        None,
        &receiver,
        crate::call::BuiltinArguments::from_borrowed(&[base, arg], None)?,
        method,
    )
    .await
}

pub async fn try_call_object_operator_method_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    method: &str,
    arg: Value,
) -> Result<OperatorMethodDispatch, RuntimeError> {
    let receiver = base.clone();
    runtime
        .scope(operator_method_dispatch(
            Some(runtime),
            &receiver,
            crate::call::BuiltinArguments::from_borrowed(&[base, arg], Some(runtime))?,
            method,
        ))
        .await
}

pub async fn call_rhs_object_operator_method_ordered(
    lhs: Value,
    rhs: Value,
    method: &str,
) -> Result<RootedValue, RuntimeError> {
    match try_call_rhs_object_operator_method_ordered(lhs, rhs, method).await? {
        Some(value) => Ok(value),
        None => Err(missing_operator_method(method)),
    }
}

pub async fn call_rhs_object_operator_method_ordered_in_context(
    runtime: &crate::context::RuntimeContext,
    lhs: Value,
    rhs: Value,
    method: &str,
) -> Result<RootedValue, RuntimeError> {
    match try_call_rhs_object_operator_method_ordered_in_context(runtime, lhs, rhs, method).await? {
        Some(value) => Ok(value),
        None => Err(missing_operator_method(method)),
    }
}

pub async fn try_call_rhs_object_operator_method_ordered(
    lhs: Value,
    rhs: Value,
    method: &str,
) -> Result<OperatorMethodDispatch, RuntimeError> {
    let receiver = rhs.clone();
    operator_method_dispatch(
        None,
        &receiver,
        crate::call::BuiltinArguments::from_borrowed(&[lhs, rhs], None)?,
        method,
    )
    .await
}

pub async fn try_call_rhs_object_operator_method_ordered_in_context(
    runtime: &crate::context::RuntimeContext,
    lhs: Value,
    rhs: Value,
    method: &str,
) -> Result<OperatorMethodDispatch, RuntimeError> {
    let receiver = rhs.clone();
    runtime
        .scope(operator_method_dispatch(
            Some(runtime),
            &receiver,
            crate::call::BuiltinArguments::from_borrowed(&[lhs, rhs], Some(runtime))?,
            method,
        ))
        .await
}

async fn operator_method_dispatch(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: &Value,
    args: crate::call::BuiltinArguments,
    method: &str,
) -> Result<OperatorMethodDispatch, RuntimeError> {
    match try_operator_method_with_outputs(runtime, receiver, args, method, 1, None).await? {
        Some(values) => super::require_single(values).map(Some),
        None => Ok(None),
    }
}
