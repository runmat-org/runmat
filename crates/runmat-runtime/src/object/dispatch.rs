use crate::call::closures::method_access_permitted;
use crate::call::descriptor::{CallableCallKind, CallableDescriptor};
use crate::call::identity::{external_qualified_identity, CallableIdentityOwner};
use crate::execution::{RootedValue, RootedValueSequence};
use crate::object::indexing::{ObjectIndexOp, ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    CallableFallbackPolicy, CallableIdentity, ClassIdentity, MethodId, QualifiedName, SymbolName,
};
use runmat_value::{Utf8Text, Value};

#[cfg(test)]
mod tests;

mod fallback;
mod super_method;
pub(crate) use super_method::call_selected_super_method;
mod operator;
pub use operator::{
    call_object_operator_method, call_object_operator_method_in_context,
    call_rhs_object_operator_method_ordered, call_rhs_object_operator_method_ordered_in_context,
    call_rhs_operator_method_ordered_with_outputs,
    call_rhs_operator_method_ordered_with_outputs_in_context, try_call_object_operator_method,
    try_call_object_operator_method_in_context, try_call_rhs_object_operator_method_ordered,
    try_call_rhs_object_operator_method_ordered_in_context, OperatorMethodDispatch,
};

fn method_member_name(identity: &CallableIdentity) -> Option<String> {
    match identity {
        CallableIdentity::DynamicName(runmat_types::SymbolName(name)) => {
            let trimmed = name.trim();
            (!trimmed.is_empty()).then_some(trimmed.to_string())
        }
        CallableIdentity::Method(runmat_types::MethodId(name)) => {
            let trimmed = name.trim();
            (!trimmed.is_empty()).then_some(trimmed.to_string())
        }
        CallableIdentity::ExternalName(runmat_types::QualifiedName(segments))
            if segments.len() == 1 && !segments[0].0.trim().is_empty() =>
        {
            Some(segments[0].0.trim().to_string())
        }
        _ => None,
    }
}

fn runtime_named_identity(name: &str) -> (CallableIdentity, CallableFallbackPolicy) {
    if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(name.trim()) {
        return (
            CallableIdentity::BoundFunction(runmat_types::FunctionId(function)),
            CallableFallbackPolicy::None,
        );
    }
    let segments: Vec<&str> = name.split('.').collect();
    if segments.len() > 1 && segments.iter().all(|segment| !segment.trim().is_empty()) {
        let qualified = QualifiedName(
            segments
                .into_iter()
                .map(|segment| SymbolName(segment.trim().to_string()))
                .collect(),
        );
        (
            CallableIdentity::ExternalName(qualified),
            CallableFallbackPolicy::ExternalBoundary,
        )
    } else {
        (
            CallableIdentity::DynamicName(SymbolName(name.trim().to_string())),
            CallableFallbackPolicy::RuntimeNameResolution,
        )
    }
}

async fn call_identity_with_policy(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: CallableIdentity,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    let descriptor = CallableDescriptor::resolved(
        identity,
        Vec::new(),
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    )
    .with_retained_arguments(args)?;
    match runtime {
        Some(runtime) => {
            crate::call::descriptor::execute_retained_callable_descriptor(Some(runtime), descriptor)
                .await
        }
        None => {
            crate::call::descriptor::execute_retained_callable_descriptor(None, descriptor).await
        }
    }
}

async fn call_registered_method(
    runtime: Option<&crate::context::RuntimeContext>,
    method: crate::class_registry::BoundRuntimeMethod,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    if let Some(binding) = method.bound_function {
        let descriptor = CallableDescriptor::bound(
            binding,
            Vec::new(),
            requested_outputs,
            CallableCallKind::Direct,
        )
        .with_retained_arguments(args)?;
        return match runtime {
            Some(runtime) => {
                crate::call::descriptor::execute_retained_callable_descriptor(
                    Some(runtime),
                    descriptor,
                )
                .await
            }
            None => {
                crate::call::descriptor::execute_retained_callable_descriptor(None, descriptor)
                    .await
            }
        };
    }
    call_identity_with_policy(
        runtime,
        method.callable,
        args,
        requested_outputs,
        method.fallback,
    )
    .await
}

async fn try_call_identity_with_policy(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: CallableIdentity,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let descriptor = CallableDescriptor::resolved(
        identity,
        Vec::new(),
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    )
    .with_retained_arguments(args)?;
    match runtime {
        Some(runtime) => {
            crate::call::descriptor::try_execute_retained_callable_descriptor(
                Some(runtime),
                descriptor,
            )
            .await
        }
        None => {
            crate::call::descriptor::try_execute_retained_callable_descriptor(None, descriptor)
                .await
        }
    }
}

async fn call_member_index_on_object_like(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: Value,
    class_name: &ClassIdentity,
    name: Utf8Text,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentityOwner>,
) -> Result<RootedValueSequence, RuntimeError> {
    if args.is_empty() {
        let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
            caller_function_name,
        );
        if let Some(method) = crate::object::protocol::resolve_object_protocol(
            &receiver,
            crate::object::protocol::ObjectProtocol::Subsref,
            &access,
        )? {
            let resolution = Some(method);
            let path = ObjectSubscriptPath::single_in_context(
                runtime,
                ObjectSubscript::member_retained(name),
            )?;
            return match runtime {
                Some(runtime) => {
                    crate::object::protocol::invoke_resolved_object_protocol_in_context(
                        runtime,
                        &resolution,
                        receiver,
                        path,
                        requested_outputs,
                    )
                    .await
                }
                None => {
                    crate::object::protocol::invoke_resolved_object_protocol(
                        &resolution,
                        receiver,
                        path,
                        requested_outputs,
                    )
                    .await
                }
            };
        }
    }
    let method_name = runmat_types::MethodName::from(name.as_str());
    if crate::object::protocol::selected_receiver_has_authority(&receiver)? {
        let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
            caller_function_name,
        );
        let method = crate::object::protocol::resolve_declared_object_method(
            &receiver,
            &method_name,
            &access,
        )?;
        let Some(method) = method else {
            if crate::object::resolve::has_stored_member(&receiver, &name.as_str().into())?
                == crate::object::resolve::MemberPresence::Present
            {
                return call_getfield_with_indices_with_context(
                    runtime,
                    receiver,
                    name,
                    args,
                    requested_outputs,
                )
                .await?
                .try_into_sequence();
            }
            return Err(semantic_error(
                "UndefinedFunction",
                "selected class has no admitted method for this member",
            ));
        };
        let full_args = prepend_argument(runtime, args, receiver)?;
        return crate::object::protocol::invoke_resolved_object_method_retained(
            runtime,
            &method,
            full_args,
            requested_outputs,
        )
        .await;
    }
    if let Some(bound) = crate::class_registry::lookup_bound_method(class_name, &method_name) {
        let m = &bound.declaration;
        if m.is_static {
            return Err(semantic_error(
                "MethodStaticOnInstance",
                format!(
                    "Method '{}' is static; use classref({}).{}",
                    name, class_name, name
                ),
            ));
        }
        if !method_access_permitted(&bound.declaring_class, &m.access, caller_function_name) {
            return Err(semantic_error(
                "MethodPrivate",
                format!("Method '{}' is private", name),
            ));
        }
        let full_args = prepend_argument(runtime, args, receiver)?;
        return call_registered_method(runtime, bound, full_args, requested_outputs).await;
    }

    let method_args = prepend_argument(runtime, args.fork(), receiver.clone())?;
    let qualified_identity = external_qualified_identity(class_name.display_name(), &name);
    if let Some(v) = try_call_identity_with_policy(
        runtime,
        qualified_identity.clone(),
        method_args.clone(),
        requested_outputs,
        CallableFallbackPolicy::ExternalBoundary,
    )
    .await?
    {
        return Ok(v);
    }
    // Prevent recursive re-entry for operator overloading (e.g. builtin `plus` calling back
    // into object dispatch). If class-qualified lookup fails, surface the miss to arithmetic
    // fallback instead of resolving unqualified operator names at runtime.
    if operator::is_operator_overload_method(&method_name) {
        return call_identity_with_policy(
            runtime,
            qualified_identity,
            method_args,
            requested_outputs,
            CallableFallbackPolicy::ExternalBoundary,
        )
        .await;
    }

    let (name_identity, name_fallback) = semantic_fallback
        .map(|identity| (identity, CallableFallbackPolicy::None))
        .unwrap_or_else(|| {
            let (identity, policy) = runtime_named_identity(&name);
            (CallableIdentityOwner::Owned(identity), policy)
        });
    if let Some(v) = fallback::try_call(
        runtime,
        name_identity,
        name.clone(),
        method_args.clone(),
        requested_outputs,
        name_fallback,
    )
    .await?
    {
        return Ok(v);
    }

    if name.as_str() == crate::OBJECT_INDEX_PAREN || name.as_str() == crate::OBJECT_INDEX_BRACE {
        return Err(semantic_error(
            "MissingSubsref",
            "class does not define subsref for indexing operation",
        ));
    }

    call_getfield_with_indices_with_context(runtime, receiver, name, args, requested_outputs)
        .await?
        .try_into_sequence()
}

pub async fn call_getfield_with_indices(
    base: Value,
    field: String,
    indices: Vec<Value>,
    _requested_outputs: usize,
) -> Result<RootedValue, RuntimeError> {
    let indices = retain_owned_arguments(None, indices)?;
    let field = crate::call::argument_text::adopt_string(field, None)?;
    call_getfield_with_indices_with_context(None, base, field, indices, _requested_outputs).await
}

async fn call_getfield_with_indices_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: Utf8Text,
    indices: crate::call::BuiltinArguments,
    _requested_outputs: usize,
) -> Result<RootedValue, RuntimeError> {
    let count = if indices.is_empty() { 1 } else { 2 };
    let getfield_args = indices
        .try_map_mixed_async(count, runtime, async |slot| {
            if slot == 0 {
                return Ok(crate::execution::ArgumentProjection::Source(Value::String(
                    field.clone().into(),
                )));
            }
            crate::execution::structure::cell::from_borrowed_values(
                runtime,
                &[1, indices.len()],
                indices.iter(),
            )
            .map(crate::execution::ArgumentProjection::Rooted)
        })
        .await?;
    crate::builtins::structs::core::getfield::getfield_internal_retained_with_context(
        runtime,
        base,
        getfield_args,
    )
    .await
}

pub async fn call_object_named_method_with_outputs(
    base: Value,
    method: String,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    call_method_or_member_index_with_outputs(
        base,
        CallableIdentity::Method(MethodId(method.clone())),
        args,
        requested_outputs,
        None,
        CallableFallbackPolicy::ObjectDispatch,
    )
    .await
}

pub async fn call_object_property_getter_with_outputs(
    base: Value,
    field: &str,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    call_object_named_method_with_outputs(
        base,
        crate::object_property_getter_name(field),
        vec![],
        requested_outputs,
    )
    .await
}

pub async fn call_object_property_getter_with_outputs_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    field: &str,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    let name = crate::call::argument_text::adopt_string(
        crate::object_property_getter_name(field),
        Some(runtime),
    )?;
    let args = crate::call::BuiltinArguments::from_borrowed(&[], Some(runtime))?;
    call_method_or_member_index_with_fallback_with_context(
        Some(runtime),
        base,
        name,
        args,
        requested_outputs,
        None,
        None,
    )
    .await
}

pub async fn call_object_property_setter_with_outputs(
    base: Value,
    field: &str,
    value: Value,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    call_object_named_method_with_outputs(
        base,
        crate::object_property_setter_name(field),
        vec![value],
        requested_outputs,
    )
    .await
}

pub async fn call_object_property_setter_with_outputs_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    field: &str,
    value: Value,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    let name = crate::call::argument_text::adopt_string(
        crate::object_property_setter_name(field),
        Some(runtime),
    )?;
    let args = crate::call::BuiltinArguments::from_borrowed(&[value], Some(runtime))?;
    call_method_or_member_index_with_fallback_with_context(
        Some(runtime),
        base,
        name,
        args,
        requested_outputs,
        None,
        None,
    )
    .await
}

async fn call_object_member_method(
    base: Value,
    op: ObjectIndexOp,
    field: String,
    rhs: Option<Value>,
) -> Result<RootedValue, RuntimeError> {
    let resolution = resolve_object_index_protocol(&base, op, None)?;
    let path = ObjectSubscriptPath::single(ObjectSubscript::member(field));
    match op {
        ObjectIndexOp::Subsref => require_single(
            invoke_resolved_object_index_path_method(&resolution, base, path, 1).await?,
        ),
        ObjectIndexOp::Subsasgn => {
            let Some(method) = resolution else {
                return Err(semantic_error(
                    "MissingObjectProtocol",
                    "object does not define subsasgn",
                ));
            };
            let rhs = rhs.ok_or_else(|| {
                semantic_error(
                    "InvalidObjectAssignment",
                    "object assignment is missing its RHS",
                )
            })?;
            crate::object::protocol::invoke_resolved_object_assignment(
                &method,
                base,
                path,
                vec![rhs],
            )
            .await
        }
    }
}

pub async fn call_object_member_subsref(
    base: Value,
    field: String,
) -> Result<RootedValue, RuntimeError> {
    call_object_member_method(base, ObjectIndexOp::Subsref, field, None).await
}

pub async fn call_object_member_subsasgn(
    base: Value,
    field: String,
    rhs: Value,
) -> Result<RootedValue, RuntimeError> {
    call_object_member_method(base, ObjectIndexOp::Subsasgn, field, Some(rhs)).await
}

pub fn resolve_object_index_protocol(
    base: &Value,
    op: ObjectIndexOp,
    caller_function_name: Option<&str>,
) -> Result<crate::object::protocol::ProtocolResolution, RuntimeError> {
    resolve_object_index_protocol_with_access(base, op, caller_function_name, None)
}

pub(crate) fn resolve_object_index_protocol_with_access(
    base: &Value,
    op: ObjectIndexOp,
    caller_function_name: Option<&str>,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
) -> Result<crate::object::protocol::ProtocolResolution, RuntimeError> {
    let protocol = match op {
        ObjectIndexOp::Subsref => crate::object::protocol::ObjectProtocol::Subsref,
        ObjectIndexOp::Subsasgn => crate::object::protocol::ObjectProtocol::Subsasgn,
    };
    match access {
        Some(access) => crate::object::protocol::resolve_object_protocol(base, protocol, access),
        None => {
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            crate::object::protocol::resolve_object_protocol(base, protocol, &access)
        }
    }
}

pub async fn invoke_resolved_object_index_path_method(
    resolution: &crate::object::protocol::ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    crate::object::protocol::invoke_resolved_object_protocol(
        resolution,
        base,
        path,
        requested_outputs,
    )
    .await
}

pub async fn invoke_resolved_object_index_path_method_in_context(
    runtime: &crate::context::RuntimeContext,
    resolution: &crate::object::protocol::ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    crate::object::protocol::invoke_resolved_object_protocol_in_context(
        runtime,
        resolution,
        base,
        path,
        requested_outputs,
    )
    .await
}

pub async fn call_method_or_member_index_with_outputs(
    base: Value,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    _fallback_policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    let name = method_member_name(&identity).ok_or_else(|| {
        semantic_error(
            "MethodCallCalleeInvalid",
            format!(
                "method/member-index call requires method-like callable identity, got {identity:?}"
            ),
        )
    })?;
    Box::pin(call_method_or_member_index_named_with_outputs(
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
    ))
    .await
}

pub async fn call_method_or_member_index_with_outputs_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    _fallback_policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    let name = method_member_name(&identity).ok_or_else(|| {
        semantic_error(
            "MethodCallCalleeInvalid",
            format!(
                "method/member-index call requires method-like callable identity, got {identity:?}"
            ),
        )
    })?;
    call_method_or_member_index_with_fallback_in_context(
        runtime,
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        None,
    )
    .await
}

pub async fn call_method_or_member_index_named_with_outputs(
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    call_method_or_member_index_with_fallback(
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        None,
    )
    .await
}

/// A semantic function identity is a fallback after receiver method dispatch,
/// never a replacement for a declared method or stored member.
pub async fn call_method_or_member_index_with_fallback(
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentity>,
) -> Result<RootedValueSequence, RuntimeError> {
    let args = retain_owned_arguments(None, args)?;
    let name = crate::call::argument_text::adopt_string(name, None)?;
    call_method_or_member_index_with_fallback_with_context(
        None,
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        semantic_fallback.map(CallableIdentityOwner::Owned),
    )
    .await
}

pub async fn call_method_or_member_index_with_fallback_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentity>,
) -> Result<RootedValueSequence, RuntimeError> {
    let args = retain_owned_arguments(Some(runtime), args)?;
    let name = crate::call::argument_text::adopt_string(name, Some(runtime))?;
    call_method_or_member_index_with_fallback_with_context(
        Some(runtime),
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        semantic_fallback.map(CallableIdentityOwner::Owned),
    )
    .await
}

async fn call_method_or_member_index_with_fallback_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: Utf8Text,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentityOwner>,
) -> Result<RootedValueSequence, RuntimeError> {
    let work = call_method_or_member_index_with_fallback_in_scope(
        runtime,
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        semantic_fallback,
    );
    match runtime {
        Some(runtime) => runtime.scope(work).await,
        None => work.await,
    }
}

/// The compiled path retains its original immutable identity and text owners.
/// Ordinary callers continue to use the owned-name public entrypoints above.
pub(crate) async fn call_method_or_member_index_retained_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: Utf8Text,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<std::sync::Arc<CallableIdentity>>,
) -> Result<RootedValueSequence, RuntimeError> {
    call_method_or_member_index_with_fallback_with_context(
        runtime,
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        semantic_fallback.map(CallableIdentityOwner::Retained),
    )
    .await
}

async fn call_method_or_member_index_with_fallback_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: Utf8Text,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentityOwner>,
) -> Result<RootedValueSequence, RuntimeError> {
    match base {
        Value::Object(obj) => {
            let class_name = obj.class_name().clone();
            Box::pin(call_member_index_on_object_like(
                runtime,
                Value::Object(obj),
                &class_name,
                name,
                args,
                requested_outputs,
                caller_function_name,
                semantic_fallback,
            ))
            .await
        }
        Value::ObjectArray(array) => {
            let class_name = array.class_name().clone();
            Box::pin(call_member_index_on_object_like(
                runtime,
                Value::ObjectArray(array),
                &class_name,
                name,
                args,
                requested_outputs,
                caller_function_name,
                semantic_fallback,
            ))
            .await
        }
        Value::HandleObject(handle) => {
            let class_name = handle.class_name.as_identity().clone();
            Box::pin(call_member_index_on_object_like(
                runtime,
                Value::HandleObject(handle),
                &class_name,
                name,
                args,
                requested_outputs,
                caller_function_name,
                semantic_fallback,
            ))
            .await
        }
        Value::ClassRef(cls) => {
            let method_name = runmat_types::MethodName::from(name.as_str());
            if let Some((m, owner)) = crate::class_registry::lookup_method(&cls, &method_name) {
                if !m.is_static {
                    return Err(semantic_error(
                        "MethodNotStatic",
                        format!("Method '{}' is not static", name),
                    ));
                }
                if !method_access_permitted(&owner, &m.access, caller_function_name) {
                    return Err(semantic_error(
                        "MethodPrivate",
                        format!("Method '{}' is private", name),
                    ));
                }
                if let Some(bound) = crate::class_registry::lookup_bound_method(&cls, &method_name)
                    .filter(|bound| bound.bound_function.is_some())
                {
                    return call_registered_method(runtime, bound, args, requested_outputs).await;
                }
                let (identity, fallback_policy) = runtime_named_identity(&m.function_name);
                return call_identity_with_policy(
                    runtime,
                    identity,
                    args,
                    requested_outputs,
                    fallback_policy,
                )
                .await;
            }
            if crate::class_registry::get_class(&cls).is_none() {
                return Err(semantic_error(
                    "UndefinedFunction",
                    format!("Undefined function in direct call: {cls}.{name}"),
                ));
            }

            let qualified_identity = external_qualified_identity(cls.display_name(), &name);
            call_identity_with_policy(
                runtime,
                qualified_identity,
                args,
                requested_outputs,
                CallableFallbackPolicy::ExternalBoundary,
            )
            .await
        }
        Value::Foreign(reference) => {
            crate::foreign::invoke_foreign_method_retained_text(
                runtime,
                reference,
                name,
                args,
                requested_outputs,
            )
            .await
        }
        other => {
            if let Some(identity) = semantic_fallback {
                let full_args = prepend_argument(runtime, args, other)?;
                return fallback::call(
                    runtime,
                    identity,
                    name,
                    full_args,
                    requested_outputs,
                    CallableFallbackPolicy::None,
                )
                .await;
            }
            call_getfield_with_indices_with_context(runtime, other, name, args, requested_outputs)
                .await?
                .try_into_sequence()
        }
    }
}

fn require_single(sequence: RootedValueSequence) -> Result<RootedValue, RuntimeError> {
    sequence.try_into_single()
}

fn retain_owned_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> Result<crate::call::BuiltinArguments, RuntimeError> {
    match runtime {
        Some(runtime) => crate::execution::RootedValues::in_context(args, runtime),
        None => crate::execution::RootedValues::new(args).map_err(|e| e.into_runtime_error()),
    }
    .map(Into::into)
}
fn prepend_argument(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
    value: Value,
) -> Result<crate::call::BuiltinArguments, RuntimeError> {
    args.try_insert(
        0,
        RootedValue::in_optional_context(value, runtime)?,
        runtime,
    )
}
