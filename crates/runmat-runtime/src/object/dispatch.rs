use crate::call::closures::method_access_permitted;
use crate::call::descriptor::{
    execute_callable_descriptor, execute_callable_descriptor_in_context,
    try_execute_callable_descriptor, try_execute_callable_descriptor_in_context, CallableCallKind,
    CallableDescriptor,
};
use crate::call::identity::external_qualified_identity;
use crate::execution::{RootedValue, RootedValueSequence};
use crate::object::indexing::{ObjectIndexOp, ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    CallableFallbackPolicy, CallableIdentity, ClassIdentity, MethodId, QualifiedName, SymbolName,
};
use runmat_value::Value;

#[cfg(test)]
mod tests;

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
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    let descriptor = CallableDescriptor::resolved(
        identity,
        args,
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    );
    match runtime {
        Some(runtime) => execute_callable_descriptor_in_context(runtime, descriptor).await,
        None => execute_callable_descriptor(descriptor).await,
    }
}

async fn call_registered_method(
    runtime: Option<&crate::context::RuntimeContext>,
    method: crate::class_registry::BoundRuntimeMethod,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    if let Some(binding) = method.bound_function {
        let descriptor =
            CallableDescriptor::bound(binding, args, requested_outputs, CallableCallKind::Direct);
        return match runtime {
            Some(runtime) => execute_callable_descriptor_in_context(runtime, descriptor).await,
            None => execute_callable_descriptor(descriptor).await,
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
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let descriptor = CallableDescriptor::resolved(
        identity,
        args,
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    );
    match runtime {
        Some(runtime) => try_execute_callable_descriptor_in_context(runtime, descriptor).await,
        None => try_execute_callable_descriptor(descriptor).await,
    }
}

async fn call_member_index_on_object_like(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: Value,
    class_name: &ClassIdentity,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentity>,
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
            let path = ObjectSubscriptPath::single(ObjectSubscript::member(name));
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
                .try_map_owned(
                    |value| {
                        runmat_value::ValueSequence::single(value)
                            .map_err(crate::sequence::sequence_error_to_runtime)
                    },
                    |error| error.into_runtime_error(),
                );
            }
            return Err(semantic_error(
                "UndefinedFunction",
                "selected class has no admitted method for this member",
            ));
        };
        let mut full_args = Vec::with_capacity(1 + args.len());
        full_args.push(receiver);
        full_args.extend(args);
        return match runtime {
            Some(runtime) => {
                crate::object::protocol::invoke_resolved_object_method_in_context(
                    runtime,
                    &method,
                    full_args,
                    requested_outputs,
                )
                .await
            }
            None => {
                crate::object::protocol::invoke_resolved_object_method(
                    &method,
                    full_args,
                    requested_outputs,
                )
                .await
            }
        };
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
        let mut full_args = Vec::with_capacity(1 + args.len());
        full_args.push(receiver.clone());
        full_args.extend(args.iter().cloned());
        return call_registered_method(runtime, bound, full_args, requested_outputs).await;
    }

    let mut method_args = Vec::with_capacity(1 + args.len());
    method_args.push(receiver.clone());
    method_args.extend(args.iter().cloned());
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
        .unwrap_or_else(|| runtime_named_identity(&name));
    if let Some(v) = try_call_identity_with_policy(
        runtime,
        name_identity.clone(),
        method_args.clone(),
        requested_outputs,
        name_fallback,
    )
    .await?
    {
        return Ok(v);
    }

    if name == crate::OBJECT_INDEX_PAREN || name == crate::OBJECT_INDEX_BRACE {
        return Err(semantic_error(
            "MissingSubsref",
            "class does not define subsref for indexing operation",
        ));
    }

    call_getfield_with_indices_with_context(runtime, receiver, name, args, requested_outputs)
        .await?
        .try_map_owned(
            |value| {
                runmat_value::ValueSequence::single(value)
                    .map_err(crate::sequence::sequence_error_to_runtime)
            },
            |error| error.into_runtime_error(),
        )
}

pub async fn call_getfield_with_indices(
    base: Value,
    field: String,
    indices: Vec<Value>,
    _requested_outputs: usize,
) -> Result<RootedValue, RuntimeError> {
    call_getfield_with_indices_with_context(None, base, field, indices, _requested_outputs).await
}

async fn call_getfield_with_indices_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    indices: Vec<Value>,
    _requested_outputs: usize,
) -> Result<RootedValue, RuntimeError> {
    let mut getfield_args = Vec::with_capacity(2);
    getfield_args.push(Value::String(field));
    if !indices.is_empty() {
        let idx_count = indices.len();
        let idx_cell = build_cell_array_with_shape(indices, 1, idx_count, "getfield idx build")?;
        getfield_args.push(Value::Cell(idx_cell));
    }
    crate::builtins::structs::core::getfield::getfield_internal_with_context(
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
    call_method_or_member_index_with_outputs_in_context(
        runtime,
        base,
        CallableIdentity::Method(MethodId(crate::object_property_getter_name(field))),
        vec![],
        requested_outputs,
        None,
        CallableFallbackPolicy::ObjectDispatch,
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
    call_method_or_member_index_with_outputs_in_context(
        runtime,
        base,
        CallableIdentity::Method(MethodId(crate::object_property_setter_name(field))),
        vec![value],
        requested_outputs,
        None,
        CallableFallbackPolicy::ObjectDispatch,
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
    let protocol = match op {
        ObjectIndexOp::Subsref => crate::object::protocol::ObjectProtocol::Subsref,
        ObjectIndexOp::Subsasgn => crate::object::protocol::ObjectProtocol::Subsasgn,
    };
    let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
        caller_function_name,
    );
    crate::object::protocol::resolve_object_protocol(base, protocol, &access)
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
    call_method_or_member_index_with_fallback_with_context(
        None,
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        semantic_fallback,
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
    call_method_or_member_index_with_fallback_with_context(
        Some(runtime),
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
        semantic_fallback,
    )
    .await
}

async fn call_method_or_member_index_with_fallback_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentity>,
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

async fn call_method_or_member_index_with_fallback_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    semantic_fallback: Option<CallableIdentity>,
) -> Result<RootedValueSequence, RuntimeError> {
    match base {
        Value::Object(obj) => {
            let class_name = obj.class_name.clone();
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
            let class_name = handle.class_name.clone();
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
            crate::foreign::invoke_foreign_method(reference, name, args, requested_outputs).await
        }
        other => {
            if let Some(identity) = semantic_fallback {
                let mut full_args = Vec::with_capacity(1 + args.len());
                full_args.push(other);
                full_args.extend(args);
                return call_identity_with_policy(
                    runtime,
                    identity,
                    full_args,
                    requested_outputs,
                    CallableFallbackPolicy::None,
                )
                .await;
            }
            call_getfield_with_indices_with_context(runtime, other, name, args, requested_outputs)
                .await?
                .try_map_owned(
                    |value| {
                        runmat_value::ValueSequence::single(value)
                            .map_err(crate::sequence::sequence_error_to_runtime)
                    },
                    |error| error.into_runtime_error(),
                )
        }
    }
}

fn require_single(sequence: RootedValueSequence) -> Result<RootedValue, RuntimeError> {
    use crate::sequence::ResolveValueSequence;
    sequence.try_map_owned(
        |sequence| {
            let values = sequence.resolve(
                runmat_types::SequenceUse::RequireSingle,
                crate::sequence::SequenceResolutionContext::default(),
            )?;
            crate::sequence::into_first_output(values)?.ok_or_else(|| {
                semantic_error(
                    "CommaSeparatedListRequiresSingleValue",
                    "object operation requires one value",
                )
            })
        },
        |error| error.into_runtime_error(),
    )
}

fn build_cell_array_with_shape(
    values: Vec<Value>,
    rows: usize,
    cols: usize,
    context: &str,
) -> Result<runmat_value::CellArray, RuntimeError> {
    runmat_value::CellArray::new(values, rows, cols)
        .map_err(|error| semantic_error("ShapeMismatch", format!("{context}: {error}")))
}
