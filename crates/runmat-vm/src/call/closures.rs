use crate::call::descriptor::{
    execute_callable_descriptor, try_execute_callable_descriptor, CallableCallKind,
    CallableDescriptor,
};
use crate::call::shared::{
    call_getfield_with_indices, external_qualified_display_name, external_qualified_identity,
};
use crate::interpreter::errors::mex;
use crate::interpreter::stack::{pop_args, pop_value};
use runmat_builtins::{builtin_functions, lookup_method, Access, Closure, Value};
use runmat_hir::{CallableFallbackPolicy, CallableIdentity, MethodId};
use runmat_runtime::RuntimeError;

fn method_identity(name: String) -> CallableIdentity {
    CallableIdentity::Method(MethodId(name))
}

fn method_member_name(identity: &CallableIdentity) -> Option<String> {
    match identity {
        CallableIdentity::DynamicName(runmat_hir::SymbolName(name)) => {
            (!name.is_empty()).then_some(name.clone())
        }
        CallableIdentity::Method(runmat_hir::MethodId(name)) => {
            (!name.is_empty()).then_some(name.clone())
        }
        CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
            if segments.len() == 1 && !segments[0].0.is_empty() =>
        {
            Some(segments[0].0.clone())
        }
        _ => None,
    }
}

async fn call_identity_with_policy(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    execute_callable_descriptor(CallableDescriptor::resolved(
        identity,
        args,
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    ))
    .await
}

async fn try_call_identity_with_policy(
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Option<Value>, RuntimeError> {
    try_execute_callable_descriptor(CallableDescriptor::resolved(
        identity,
        args,
        requested_outputs,
        fallback_policy,
        CallableCallKind::Direct,
    ))
    .await
}

async fn call_member_index_on_object_like(
    receiver: Value,
    class_name: &str,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    let post_object_fallback = fallback_policy.post_object_dispatch();
    if let Some((m, _owner)) = lookup_method(class_name, &name) {
        if m.is_static {
            return Err(format!(
                "Method '{}' is static; use classref({}).{}",
                name, class_name, name
            )
            .into());
        }
        if m.access == Access::Private {
            return Err(format!("Method '{}' is private", name).into());
        }
        let mut full_args = Vec::with_capacity(1 + args.len());
        full_args.push(receiver.clone());
        full_args.extend(args.iter().cloned());
        return call_identity_with_policy(
            method_identity(m.function_name.clone()),
            full_args,
            requested_outputs,
            CallableFallbackPolicy::RuntimeNameResolution,
        )
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
    if let Some(v) = try_call_identity_with_policy(
        method_identity(name.clone()),
        method_args.clone(),
        requested_outputs,
        CallableFallbackPolicy::ObjectDispatch,
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

    match call_identity_with_policy(
        method_identity(name.clone()),
        method_args,
        requested_outputs,
        post_object_fallback,
    )
    .await
    {
        Ok(v) => return Ok(v),
        Err(err) if err.identifier() == Some("RunMat:UndefinedFunction") => {}
        Err(err) => return Err(err),
    }

    call_getfield_with_indices(receiver, name, args, requested_outputs).await
}

pub fn create_closure(
    stack: &mut Vec<Value>,
    func_name: String,
    capture_count: usize,
) -> Result<(), RuntimeError> {
    let mut captures = Vec::with_capacity(capture_count);
    for _ in 0..capture_count {
        captures.push(pop_value(stack)?);
    }
    captures.reverse();
    stack.push(Value::Closure(Closure {
        function_name: func_name,
        semantic_function: None,
        captures,
    }));
    Ok(())
}

pub fn create_semantic_closure(
    stack: &mut Vec<Value>,
    function: runmat_hir::FunctionId,
    display_name: String,
    capture_count: usize,
) -> Result<(), RuntimeError> {
    let mut captures = Vec::with_capacity(capture_count);
    for _ in 0..capture_count {
        captures.push(pop_value(stack)?);
    }
    captures.reverse();
    stack.push(Value::Closure(Closure {
        function_name: display_name,
        semantic_function: Some(function.0),
        captures,
    }));
    Ok(())
}

pub fn load_method_closure(base: Value, name: String) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            let function_name = external_qualified_display_name(&obj.class_name, &name);
            Ok(Value::Closure(Closure {
                semantic_function:
                    runmat_runtime::user_functions::resolve_semantic_function_by_name(
                        &function_name,
                    ),
                function_name,
                captures: vec![Value::Object(obj)],
            }))
        }
        Value::ClassRef(cls) => {
            if let Some((m, _owner)) = lookup_method(&cls, &name) {
                if !m.is_static {
                    return Err(format!("Method '{}' is not static", name).into());
                }
                return Ok(Value::Closure(Closure {
                    semantic_function:
                        runmat_runtime::user_functions::resolve_semantic_function_by_name(
                            &m.function_name,
                        ),
                    function_name: m.function_name,
                    captures: vec![],
                }));
            }
            let qualified_name = external_qualified_display_name(&cls, &name);
            if builtin_functions().iter().any(|b| b.name == qualified_name) {
                Ok(Value::Closure(Closure {
                    semantic_function:
                        runmat_runtime::user_functions::resolve_semantic_function_by_name(
                            &qualified_name,
                        ),
                    function_name: qualified_name,
                    captures: vec![],
                }))
            } else {
                Err(format!("Unknown static method '{}' on class {}", name, cls).into())
            }
        }
        _ => Err(mex("LoadMethod", "LoadMethod requires object or classref")),
    }
}

pub async fn call_method_or_member_index_with_outputs(
    base: Value,
    identity: CallableIdentity,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
) -> Result<Value, RuntimeError> {
    let name = method_member_name(&identity).ok_or_else(|| {
        mex(
            "UndefinedFunction",
            "method/member-index call missing callable name",
        )
    })?;
    match base {
        Value::Object(obj) => {
            let class_name = obj.class_name.clone();
            call_member_index_on_object_like(
                Value::Object(obj),
                &class_name,
                name,
                args,
                requested_outputs,
                fallback_policy,
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
                fallback_policy,
            )
            .await
        }
        Value::ClassRef(cls) => {
            if let Some((m, _owner)) = lookup_method(&cls, &name) {
                if !m.is_static {
                    return Err(format!("Method '{}' is not static", name).into());
                }
                return call_identity_with_policy(
                    method_identity(m.function_name.clone()),
                    args,
                    requested_outputs,
                    CallableFallbackPolicy::RuntimeNameResolution,
                )
                .await;
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

#[cfg(test)]
mod tests {
    use super::call_method_or_member_index_with_outputs;
    use futures::executor::block_on;
    use runmat_builtins::Value;
    use runmat_hir::{CallableFallbackPolicy, CallableIdentity, MethodId};
    use std::sync::Arc;

    #[test]
    fn classref_external_method_uses_external_boundary_semantic_resolution() {
        let _resolver_guard =
            runmat_runtime::user_functions::install_semantic_function_resolver(Some(Arc::new(
                |name| (name == "Point.remote_inc").then_some(7331),
            )));
        let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(
            Some(Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 7331);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(2.0)]);
                Box::pin(async { Ok(Value::Num(3.0)) })
            })),
        );
        let value = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::Method(MethodId("remote_inc".to_string())),
            vec![Value::Num(2.0)],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect("classref external call should resolve through semantic resolver");
        assert_eq!(value, Value::Num(3.0));
    }

    #[test]
    fn classref_external_method_without_resolver_remains_unresolved() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::Method(MethodId("sqrt".to_string())),
            vec![Value::Num(9.0)],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("classref external call should not fallback to builtin name resolution");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn method_member_call_rejects_identity_without_method_name() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::AnonymousFunction(runmat_hir::FunctionId(12)),
            vec![Value::Num(9.0)],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("anonymous identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }
}

pub fn collect_method_args(
    stack: &mut Vec<Value>,
    arg_count: usize,
) -> Result<(Value, Vec<Value>), RuntimeError> {
    let args = pop_args(stack, arg_count)?;
    let base = pop_value(stack)?;
    Ok((base, args))
}
