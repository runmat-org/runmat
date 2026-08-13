use crate::call::closures::{
    caller_class_for_function, method_access_permitted, resolve_method_semantic_function_id,
};
use crate::call::descriptor::{
    execute_callable_descriptor, try_execute_callable_descriptor, CallableCallKind,
    CallableDescriptor,
};
use crate::call::identity::external_qualified_identity;
use crate::object::indexing::{
    build_matlab_substruct_arg, class_name_from_base, ObjectIndexDescriptor, ObjectIndexOp,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{CallableFallbackPolicy, CallableIdentity, MethodId, QualifiedName, SymbolName};
use runmat_value::Value;

fn caller_has_internal_class_access(caller_function_name: Option<&str>, class_name: &str) -> bool {
    caller_class_for_function(caller_function_name).is_some_and(|caller_class| {
        crate::class_registry::is_class_or_subclass(&caller_class, class_name)
            || crate::class_registry::is_class_or_subclass(class_name, &caller_class)
    })
}

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

fn method_function_identity(
    owner: &str,
    method_name: &str,
    function_name: &str,
) -> (CallableIdentity, CallableFallbackPolicy) {
    let trimmed = function_name.trim();
    if let Some(function) = resolve_method_semantic_function_id(owner, method_name, trimmed) {
        return (
            CallableIdentity::BoundFunction(runmat_types::FunctionId(function)),
            CallableFallbackPolicy::None,
        );
    }
    if !trimmed.is_empty() && runmat_builtins::builtin_name_is_known(trimmed) {
        return runtime_named_identity(trimmed);
    }
    if trimmed.is_empty() {
        return (
            external_qualified_identity(owner, method_name),
            CallableFallbackPolicy::ExternalBoundary,
        );
    }
    if trimmed.contains('.') {
        return runtime_named_identity(trimmed);
    }
    (
        external_qualified_identity(owner, trimmed),
        CallableFallbackPolicy::ExternalBoundary,
    )
}

fn is_operator_overload_name(name: &str) -> bool {
    matches!(
        name,
        "plus"
            | "minus"
            | "times"
            | "mtimes"
            | "rdivide"
            | "mrdivide"
            | "ldivide"
            | "mldivide"
            | "power"
            | "mpower"
            | "uminus"
            | "uplus"
            | "lt"
            | "le"
            | "gt"
            | "ge"
            | "eq"
            | "ne"
            | "and"
            | "or"
            | "xor"
            | "not"
    )
}

fn is_receiver_validation_error(err: &RuntimeError) -> bool {
    err.identifier()
        .is_some_and(|identifier| identifier.ends_with("ReceiverInvalid"))
}

async fn call_identity_with_policy(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    Box::pin(execute_callable_descriptor(CallableDescriptor::resolved(
        identity,
        args,
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    )))
    .await
}

async fn try_call_identity_with_policy(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<Value>, RuntimeError> {
    Box::pin(try_execute_callable_descriptor(
        CallableDescriptor::resolved(
            identity,
            args,
            requested_outputs,
            fallback_policy,
            CallableCallKind::Direct,
        ),
    ))
    .await
}

async fn call_member_index_on_object_like(
    receiver: Value,
    class_name: &str,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    if args.is_empty()
        && crate::class_registry::get_class(class_name)
            .is_some_and(|class_def| class_defines_member_subsref(&class_def))
        && !caller_has_internal_class_access(caller_function_name, class_name)
    {
        return Box::pin(call_object_member_subsref(receiver, name)).await;
    }
    if let Some((m, owner)) = crate::class_registry::lookup_method(class_name, &name) {
        if m.is_static {
            return Err(semantic_error(
                "MethodStaticOnInstance",
                format!(
                    "Method '{}' is static; use classref({}).{}",
                    name, class_name, name
                ),
            ));
        }
        if !method_access_permitted(&owner, &m.access, caller_function_name) {
            return Err(semantic_error(
                "MethodPrivate",
                format!("Method '{}' is private", name),
            ));
        }
        let mut full_args = Vec::with_capacity(1 + args.len());
        full_args.push(receiver.clone());
        full_args.extend(args.iter().cloned());
        let (identity, fallback_policy) = method_function_identity(&owner, &name, &m.function_name);
        return call_identity_with_policy(identity, full_args, requested_outputs, fallback_policy)
            .await;
    }

    let mut method_args = Vec::with_capacity(1 + args.len());
    method_args.push(receiver.clone());
    method_args.extend(args.iter().cloned());
    let qualified_identity = external_qualified_identity(class_name, &name);
    if let Some(v) = try_call_identity_with_policy(
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
    if is_operator_overload_name(&name) {
        return call_identity_with_policy(
            qualified_identity,
            method_args,
            requested_outputs,
            CallableFallbackPolicy::ExternalBoundary,
        )
        .await;
    }

    let (name_identity, name_fallback) = runtime_named_identity(&name);
    if let Some(v) = try_call_identity_with_policy(
        name_identity.clone(),
        method_args.clone(),
        requested_outputs,
        name_fallback,
    )
    .await?
    {
        return Ok(v);
    }

    match call_identity_with_policy(
        qualified_identity,
        method_args.clone(),
        requested_outputs,
        CallableFallbackPolicy::ExternalBoundary,
    )
    .await
    {
        Ok(v) => return Ok(v),
        Err(err) if err.identifier() == Some("RunMat:UndefinedFunction") => {}
        Err(err) => return Err(err),
    }

    match call_identity_with_policy(name_identity, method_args, requested_outputs, name_fallback)
        .await
    {
        Ok(v) => return Ok(v),
        Err(err) if err.identifier() == Some("RunMat:UndefinedFunction") => {}
        Err(err) => return Err(err),
    }

    if name == crate::OBJECT_INDEX_PAREN || name == crate::OBJECT_INDEX_BRACE {
        return Err(semantic_error(
            "MissingSubsref",
            "class does not define subsref for indexing operation",
        ));
    }

    call_getfield_with_indices(receiver, name, args, requested_outputs).await
}

pub async fn call_rhs_operator_method_ordered_with_outputs(
    lhs: Value,
    rhs: Value,
    name: String,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    let class_name = match &rhs {
        Value::Object(obj) => obj.class_name.clone(),
        Value::HandleObject(handle) => handle.class_name.clone(),
        _ => {
            return Err(semantic_error(
                "InvalidObjectDispatch",
                "right-hand operator dispatch requires an object operand",
            ));
        }
    };

    let method_args = vec![lhs.clone(), rhs.clone()];
    if let Some((m, owner)) = crate::class_registry::lookup_method(&class_name, &name) {
        if m.is_static {
            return Err(semantic_error(
                "MethodStaticOnInstance",
                format!(
                    "Method '{}' is static; use classref({}).{}",
                    name, class_name, name
                ),
            ));
        }
        if !method_access_permitted(&owner, &m.access, caller_function_name) {
            return Err(semantic_error(
                "MethodPrivate",
                format!("Method '{}' is private", name),
            ));
        }
        let (identity, fallback_policy) = method_function_identity(&owner, &name, &m.function_name);
        return match call_identity_with_policy(
            identity.clone(),
            method_args,
            requested_outputs,
            fallback_policy,
        )
        .await
        {
            Ok(v) => Ok(v),
            Err(err) if is_receiver_validation_error(&err) && is_operator_overload_name(&name) => {
                call_identity_with_policy(
                    identity,
                    vec![rhs.clone(), lhs.clone()],
                    requested_outputs,
                    fallback_policy,
                )
                .await
            }
            Err(err) => Err(err),
        };
    }

    let qualified_identity = external_qualified_identity(&class_name, &name);
    let ordered_result = call_identity_with_policy(
        qualified_identity.clone(),
        method_args.clone(),
        requested_outputs,
        CallableFallbackPolicy::ExternalBoundary,
    )
    .await;
    match ordered_result {
        Ok(v) => Ok(v),
        Err(ordered_err) => {
            if ordered_err.identifier() != Some("RunMat:UndefinedFunction")
                && !is_receiver_validation_error(&ordered_err)
            {
                return Err(ordered_err);
            }
            let receiver_first_args = vec![rhs.clone(), lhs.clone()];
            match call_identity_with_policy(
                qualified_identity.clone(),
                receiver_first_args,
                requested_outputs,
                CallableFallbackPolicy::ExternalBoundary,
            )
            .await
            {
                Ok(v) => Ok(v),
                Err(receiver_err) => Err(receiver_err),
            }
        }
    }
}

pub async fn call_getfield_with_indices(
    base: Value,
    field: String,
    indices: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    let mut getfield_args = Vec::with_capacity(3);
    getfield_args.push(base);
    getfield_args.push(Value::String(field));
    if !indices.is_empty() {
        let idx_count = indices.len();
        let idx_cell = build_cell_array_with_shape(indices, 1, idx_count, "getfield idx build")?;
        getfield_args.push(Value::Cell(idx_cell));
    }
    crate::call_builtin_async_with_outputs("getfield", &getfield_args, requested_outputs).await
}

pub async fn call_object_operator_method(
    base: Value,
    method: &str,
    arg: Value,
) -> Result<Value, RuntimeError> {
    call_method_or_member_index_with_outputs(
        base,
        CallableIdentity::Method(MethodId(method.to_string())),
        vec![arg],
        1,
        None,
        CallableFallbackPolicy::ObjectDispatch,
    )
    .await
}

pub async fn call_rhs_object_operator_method_ordered(
    lhs: Value,
    rhs: Value,
    method: &str,
) -> Result<Value, RuntimeError> {
    call_rhs_operator_method_ordered_with_outputs(lhs, rhs, method.to_string(), 1, None).await
}

pub async fn call_object_named_method_with_outputs(
    base: Value,
    method: String,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
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
) -> Result<Value, RuntimeError> {
    call_object_named_method_with_outputs(
        base,
        crate::object_property_getter_name(field),
        vec![],
        requested_outputs,
    )
    .await
}

pub async fn call_object_property_setter_with_outputs(
    base: Value,
    field: &str,
    value: Value,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    call_object_named_method_with_outputs(
        base,
        crate::object_property_setter_name(field),
        vec![value],
        requested_outputs,
    )
    .await
}

async fn call_object_member_method(
    base: Value,
    op: ObjectIndexOp,
    field: String,
    rhs: Option<Value>,
) -> Result<Value, RuntimeError> {
    call_object_index_descriptor_method(ObjectIndexDescriptor::member(base, op, field, rhs)).await
}

pub async fn call_object_member_subsref(base: Value, field: String) -> Result<Value, RuntimeError> {
    call_object_member_method(base, ObjectIndexOp::Subsref, field, None).await
}

pub async fn call_object_member_subsasgn(
    base: Value,
    field: String,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    call_object_member_method(base, ObjectIndexOp::Subsasgn, field, Some(rhs)).await
}

pub fn class_defines_member_subsref(class: &crate::class_registry::RuntimeClass) -> bool {
    crate::class_registry::lookup_method(&class.name, ObjectIndexOp::Subsref.protocol_name())
        .is_some()
}

pub fn class_defines_member_subsasgn(class: &crate::class_registry::RuntimeClass) -> bool {
    crate::class_registry::lookup_method(&class.name, ObjectIndexOp::Subsasgn.protocol_name())
        .is_some()
}

pub async fn call_object_index_descriptor_method(
    descriptor: ObjectIndexDescriptor,
) -> Result<Value, RuntimeError> {
    call_object_index_descriptor_method_with_outputs(descriptor, 1).await
}

pub async fn call_object_index_descriptor_method_with_outputs(
    descriptor: ObjectIndexDescriptor,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    if let Some(class_name) = class_name_from_base(descriptor.base()) {
        if let Some((method, owner)) =
            crate::class_registry::lookup_method(class_name, descriptor.operation().protocol_name())
        {
            let mut semantic_args = vec![
                descriptor.base().clone(),
                build_matlab_substruct_arg(&descriptor)?,
            ];
            if let Some(rhs) = descriptor.rhs() {
                semantic_args.push(rhs.clone());
            }
            if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
                &method.function_name,
                &semantic_args,
                requested_outputs,
            )
            .await
            {
                return result;
            }
            let owner_qualified = format!("{}.{}", owner, descriptor.operation().protocol_name());
            if owner_qualified != method.function_name {
                if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
                    &owner_qualified,
                    &semantic_args,
                    requested_outputs,
                )
                .await
                {
                    return result;
                }
            }
        }
    }
    let (base, method, args) = descriptor.into_method_invocation()?;
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

pub async fn call_method_or_member_index_with_outputs(
    base: Value,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
    _fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    let name = method_member_name(&identity).ok_or_else(|| {
        semantic_error(
            "MethodCallCalleeInvalid",
            format!(
                "method/member-index call requires method-like callable identity, got {identity:?}"
            ),
        )
    })?;
    call_method_or_member_index_named_with_outputs(
        base,
        name,
        args,
        requested_outputs,
        caller_function_name,
    )
    .await
}

pub async fn call_method_or_member_index_named_with_outputs(
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            let class_name = obj.class_name.clone();
            call_member_index_on_object_like(
                Value::Object(obj),
                &class_name,
                name,
                args,
                requested_outputs,
                caller_function_name,
            )
            .await
        }
        Value::HandleObject(handle) => {
            let class_name = handle.class_name.clone();
            call_member_index_on_object_like(
                Value::HandleObject(handle),
                &class_name,
                name,
                args,
                requested_outputs,
                caller_function_name,
            )
            .await
        }
        Value::ClassRef(cls) => {
            if let Some((m, owner)) = crate::class_registry::lookup_method(&cls, &name) {
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
                let (identity, fallback_policy) = runtime_named_identity(&m.function_name);
                return call_identity_with_policy(
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

            let qualified_identity = external_qualified_identity(&cls, &name);
            call_identity_with_policy(
                qualified_identity,
                args,
                requested_outputs,
                CallableFallbackPolicy::ExternalBoundary,
            )
            .await
        }
        other => call_getfield_with_indices(other, name, args, requested_outputs).await,
    }
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
