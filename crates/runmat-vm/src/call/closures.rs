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
            let trimmed = name.trim();
            (!trimmed.is_empty()).then_some(trimmed.to_string())
        }
        CallableIdentity::Method(runmat_hir::MethodId(name)) => {
            let trimmed = name.trim();
            (!trimmed.is_empty()).then_some(trimmed.to_string())
        }
        CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
            if segments.len() == 1 && !segments[0].0.trim().is_empty() =>
        {
            Some(segments[0].0.trim().to_string())
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
            return Err(mex(
                "MethodStaticOnInstance",
                &format!(
                    "Method '{}' is static; use classref({}).{}",
                    name, class_name, name
                ),
            ));
        }
        if m.access == Access::Private {
            return Err(mex(
                "MethodPrivate",
                &format!("Method '{}' is private", name),
            ));
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

    if name == runmat_runtime::OBJECT_INDEX_PAREN || name == runmat_runtime::OBJECT_INDEX_BRACE {
        return Err(mex(
            "MissingSubsref",
            "class does not define subsref for indexing operation",
        ));
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
        bound_function: None,
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
        bound_function: Some(function.0),
        captures,
    }));
    Ok(())
}

pub fn load_method_closure(base: Value, name: String) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            let function_name = external_qualified_display_name(&obj.class_name, &name);
            Ok(Value::Closure(Closure {
                bound_function: runmat_runtime::user_functions::resolve_semantic_function_by_name(
                    &function_name,
                ),
                function_name,
                captures: vec![Value::Object(obj)],
            }))
        }
        Value::ClassRef(cls) => {
            if let Some((m, _owner)) = lookup_method(&cls, &name) {
                if !m.is_static {
                    return Err(mex(
                        "MethodNotStatic",
                        &format!("Method '{}' is not static", name),
                    ));
                }
                return Ok(Value::Closure(Closure {
                    bound_function:
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
                    bound_function:
                        runmat_runtime::user_functions::resolve_semantic_function_by_name(
                            &qualified_name,
                        ),
                    function_name: qualified_name,
                    captures: vec![],
                }))
            } else {
                Err(mex(
                    "UnknownStaticMethod",
                    &format!("Unknown static method '{}' on class {}", name, cls),
                ))
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
            "MethodCallCalleeInvalid",
            &format!(
                "method/member-index call requires method-like callable identity, got {identity:?}"
            ),
        )
    })?;
    call_method_or_member_index_named_with_outputs(
        base,
        name,
        args,
        requested_outputs,
        fallback_policy,
    )
    .await
}

pub(crate) async fn call_method_or_member_index_named_with_outputs(
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: usize,
    fallback_policy: CallableFallbackPolicy,
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
                    return Err(mex(
                        "MethodNotStatic",
                        &format!("Method '{}' is not static", name),
                    ));
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
    use super::{call_method_or_member_index_with_outputs, load_method_closure};
    use futures::executor::block_on;
    use runmat_builtins::{register_class, Access, ClassDef, MethodDef, Value};
    use runmat_hir::{CallableFallbackPolicy, CallableIdentity, MethodId};
    use std::collections::HashMap;
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
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_imported_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::Imported(runmat_hir::DefPath {
                package: runmat_hir::PackageName("Point".to_string()),
                module: runmat_hir::QualifiedName(vec![
                    runmat_hir::SymbolName("Point".to_string()),
                    runmat_hir::SymbolName("origin".to_string()),
                ]),
                item: vec![runmat_hir::DefPathSegment::Function(
                    runmat_hir::SymbolName("origin".to_string()),
                )],
            }),
            vec![Value::Num(9.0)],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("imported identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_multisegment_external_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ])),
            vec![Value::Num(9.0)],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("multi-segment external identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_whitespace_method_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::Method(MethodId("   ".to_string())),
            vec![Value::Num(9.0)],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("whitespace method identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_whitespace_single_segment_external_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("   ".to_string()),
            ])),
            vec![Value::Num(9.0)],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err(
            "whitespace single-segment external identity should not be used for method/member call",
        );
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn classref_nonstatic_method_reports_identifier() {
        let class_name = "ClosureMethodNotStaticTest".to_string();
        let mut methods = HashMap::new();
        methods.insert(
            "inst".to_string(),
            MethodDef {
                name: "inst".to_string(),
                is_static: false,
                access: Access::Public,
                function_name: "inst".to_string(),
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods,
        });

        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef(class_name),
            CallableIdentity::Method(MethodId("inst".to_string())),
            vec![],
            1,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("classref call to non-static method should fail");
        assert_eq!(err.identifier(), Some("RunMat:MethodNotStatic"));
    }

    #[test]
    fn load_method_unknown_static_method_reports_identifier() {
        let err = load_method_closure(
            Value::ClassRef("Point".to_string()),
            "definitely_missing_static_method".to_string(),
        )
        .expect_err("unknown static method should fail during method-handle load");
        assert_eq!(err.identifier(), Some("RunMat:UnknownStaticMethod"));
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
