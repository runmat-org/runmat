use futures::executor::block_on;
use runmat_hir::{
    BuiltinId, CallableFallbackPolicy, CallableIdentity, DefPath, DefPathSegment, FunctionId,
    MethodId, PackageName, QualifiedName, SymbolName,
};
use runmat_runtime::call::descriptor::{
    execute_callable_descriptor, try_execute_callable_descriptor, CallableCallKind,
    CallableDescriptor, CallableTarget,
};
use runmat_value::{Closure, StringArray, Tensor, Value};
use runmat_vm::FunctionRegistry;
use std::sync::Arc;

fn imported_identity(name: &str) -> CallableIdentity {
    CallableIdentity::Imported(DefPath {
        package: PackageName("pkg".to_string()),
        module: QualifiedName(vec![
            SymbolName("pkg".to_string()),
            SymbolName(name.to_string()),
        ]),
        item: vec![DefPathSegment::Function(SymbolName(name.to_string()))],
    })
}

fn method_identity(name: &str) -> CallableIdentity {
    CallableIdentity::Method(MethodId(name.to_string()))
}

#[test]
fn builtin_descriptor_uses_requested_outputs_for_multi_result_calls() {
    let input = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).expect("tensor"));
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::Builtin(BuiltinId("max".to_string())),
        vec![input],
        2,
        CallableFallbackPolicy::None,
        CallableCallKind::Direct,
    );
    let value = block_on(execute_callable_descriptor(descriptor)).expect("execute descriptor");
    match value {
        Value::OutputList(values) => assert_eq!(values.len(), 2),
        other => panic!("expected two-output list from builtin descriptor, got {other:?}"),
    }
}

#[test]
fn builtin_descriptor_uses_requested_outputs_for_zero_result_calls() {
    let args = vec![Value::Num(9.0)];
    let expected = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "sqrt", &args, 0,
    ))
    .expect("runtime builtin with explicit zero outputs");
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::Builtin(BuiltinId("sqrt".to_string())),
        args,
        0,
        CallableFallbackPolicy::None,
        CallableCallKind::Direct,
    );
    let value = block_on(execute_callable_descriptor(descriptor)).expect("execute descriptor");
    assert_eq!(value, expected);
}

#[test]
fn external_name_descriptor_does_not_fallback_to_builtin_name_resolution() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![SymbolName("sqrt".to_string())])),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("external names should remain unresolved without semantic resolution");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn external_name_descriptor_external_boundary_can_use_semantic_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.remote_inc").then_some(7777)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 7777);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![
            SymbolName("pkg".to_string()),
            SymbolName("remote_inc".to_string()),
        ])),
        vec![Value::Num(2.0)],
        1,
        CallableFallbackPolicy::ExternalBoundary,
        CallableCallKind::Direct,
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("external boundary call should resolve through semantic registry");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn external_name_descriptor_external_boundary_without_resolver_errors() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![SymbolName(
            "definitely_missing".to_string(),
        )])),
        vec![Value::Num(2.0)],
        1,
        CallableFallbackPolicy::ExternalBoundary,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("missing external boundary call should remain unresolved");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn external_name_descriptor_external_boundary_does_not_fallback_to_builtin_name_resolution() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![SymbolName("sqrt".to_string())])),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::ExternalBoundary,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("external boundary names should remain unresolved without semantic resolution");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn dynamic_name_descriptor_runtime_name_resolution_can_reach_builtin() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::DynamicName(SymbolName("sqrt".to_string())),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("dynamic runtime name resolution should reach builtin");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn imported_identity_never_falls_back_to_builtin_name_resolution() {
    let descriptor = CallableDescriptor::resolved(
        imported_identity("sqrt"),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("imported identities should not fall back to builtin name resolution");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn imported_identity_runtime_name_resolution_can_use_semantic_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.import_only").then_some(6262)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 6262);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(5.0)]);
            Box::pin(async { Ok(Value::Num(6.0)) })
        }),
    ));
    let descriptor = CallableDescriptor::resolved(
        imported_identity("import_only"),
        vec![Value::Num(5.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("imported identity should resolve through semantic resolver");
    assert_eq!(value, Value::Num(6.0));
}

#[test]
fn method_identity_runtime_name_resolution_can_use_semantic_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "method_only").then_some(9191)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 9191);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(5.0)]);
            Box::pin(async { Ok(Value::Num(6.0)) })
        }),
    ));
    let descriptor = CallableDescriptor::resolved(
        method_identity("method_only"),
        vec![Value::Num(5.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("method identity should resolve through semantic resolver");
    assert_eq!(value, Value::Num(6.0));
}

#[test]
fn method_identity_runtime_name_resolution_without_resolver_errors() {
    let descriptor = CallableDescriptor::resolved(
        method_identity("definitely_missing_method"),
        vec![Value::Num(5.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("unresolved method identity should remain undefined");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn method_identity_error_uses_typed_identity_not_fallback_name() {
    let descriptor = CallableDescriptor::resolved(
        method_identity("definitely_missing_method"),
        vec![Value::Num(5.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("unresolved method identity should remain undefined");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    assert!(
        err.message()
            .contains("Method(MethodId(\"definitely_missing_method\"))")
            && !err
                .message()
                .contains("Undefined function in direct call: definitely_missing_method"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn method_identity_never_falls_back_to_builtin_name_resolution() {
    let descriptor = CallableDescriptor::resolved(
        method_identity("sqrt"),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("method identities should not fall back to builtin name resolution");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn try_execute_dynamic_name_runtime_name_resolution_can_reach_builtin() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::DynamicName(SymbolName("sqrt".to_string())),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let value = block_on(try_execute_callable_descriptor(descriptor))
        .expect("try_execute should allow dynamic builtin fallback");
    assert_eq!(value, Some(Value::Num(3.0)));
}

#[test]
fn try_execute_imported_identity_never_falls_back_to_builtin_name_resolution() {
    let descriptor = CallableDescriptor::resolved(
        imported_identity("sqrt"),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let value = block_on(try_execute_callable_descriptor(descriptor))
        .expect("try_execute should suppress unresolved imported identities");
    assert_eq!(value, None);
}

#[test]
fn try_execute_external_boundary_single_segment_name_returns_none_without_semantic_resolution() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![SymbolName("sqrt".to_string())])),
        vec![Value::Num(9.0)],
        1,
        CallableFallbackPolicy::ExternalBoundary,
        CallableCallKind::Direct,
    );
    let value = block_on(try_execute_callable_descriptor(descriptor))
        .expect("try_execute should suppress unresolved external boundary names");
    assert_eq!(value, None);
}

#[test]
fn try_execute_external_boundary_qualified_name_can_use_semantic_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.remote_inc").then_some(9393)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 9393);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![
            SymbolName("pkg".to_string()),
            SymbolName("remote_inc".to_string()),
        ])),
        vec![Value::Num(2.0)],
        1,
        CallableFallbackPolicy::ExternalBoundary,
        CallableCallKind::Direct,
    );
    let value = block_on(try_execute_callable_descriptor(descriptor))
        .expect("try_execute should use semantic resolver for qualified external identities");
    assert_eq!(value, Some(Value::Num(3.0)));
}

#[test]
fn feval_function_handle_builtin_prefers_builtin_identity_over_runtime_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "sqrt").then_some(4242)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, _args, _requested_outputs| {
            assert_eq!(function, 4242);
            Box::pin(async { Ok(Value::Num(123.0)) })
        }),
    ));
    let descriptor = CallableDescriptor::from_feval_value(
        Value::FunctionHandle("sqrt".to_string()),
        vec![Value::Num(9.0)],
        1,
        &FunctionRegistry::default(),
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("builtin handle feval should execute");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn feval_function_handle_qualified_name_classifies_as_external_boundary() {
    let descriptor = CallableDescriptor::from_feval_value(
        Value::FunctionHandle("pkg.remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let CallableTarget::Resolved {
        identity,
        fallback_policy,
    } = &descriptor.target
    else {
        panic!("expected resolved target");
    };
    assert!(matches!(identity, CallableIdentity::ExternalName(_)));
    assert_eq!(*fallback_policy, CallableFallbackPolicy::ExternalBoundary);
}

#[test]
fn feval_method_function_handle_classifies_as_method_identity() {
    let descriptor = CallableDescriptor::from_feval_value(
        Value::MethodFunctionHandle("resolved_method".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let CallableTarget::Resolved {
        identity,
        fallback_policy,
    } = &descriptor.target
    else {
        panic!("expected resolved target");
    };
    assert!(matches!(identity, CallableIdentity::Method(_)));
    assert_eq!(
        *fallback_policy,
        CallableFallbackPolicy::RuntimeNameResolution
    );
}

#[test]
fn feval_method_function_handle_runtime_name_resolution_can_use_semantic_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "resolved_method").then_some(5252)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 5252);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::MethodFunctionHandle("resolved_method".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("method function handle should resolve through semantic resolver");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn feval_function_handle_external_boundary_can_use_semantic_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.remote_inc").then_some(5151)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 5151);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::FunctionHandle("pkg.remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("qualified function handle should resolve through semantic resolver");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn feval_closure_without_embedded_semantic_uses_registry_name_resolution() {
    let mut registry = FunctionRegistry::default();
    registry.names.insert("inc".to_string(), FunctionId(4242));

    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 4242);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(10.0), Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(12.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::Closure(Closure {
            function_name: "inc".to_string(),
            bound_function: None,
            captures: vec![Value::Num(10.0)],
        }),
        vec![Value::Num(2.0)],
        1,
        &registry,
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("closure name should resolve through semantic registry");
    assert_eq!(value, Value::Num(12.0));
}

#[test]
fn feval_closure_with_embedded_semantic_prefers_embedded_identity() {
    let mut registry = FunctionRegistry::default();
    registry.names.insert("inc".to_string(), FunctionId(9999));

    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 4242);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(10.0), Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(12.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::Closure(Closure {
            function_name: "inc".to_string(),
            bound_function: Some(4242),
            captures: vec![Value::Num(10.0)],
        }),
        vec![Value::Num(2.0)],
        1,
        &registry,
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("embedded semantic identity should take precedence");
    assert_eq!(value, Value::Num(12.0));
}

#[test]
fn malformed_qualified_function_handle_remains_dynamic_name() {
    let descriptor = CallableDescriptor::from_feval_value(
        Value::FunctionHandle("pkg..remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let CallableTarget::Resolved {
        identity,
        fallback_policy,
    } = &descriptor.target
    else {
        panic!("expected resolved target");
    };
    assert!(matches!(
        identity,
        CallableIdentity::DynamicName(SymbolName(name)) if name == "pkg..remote_inc"
    ));
    assert_eq!(
        *fallback_policy,
        CallableFallbackPolicy::RuntimeNameResolution
    );
}

#[test]
fn malformed_qualified_external_function_handle_remains_dynamic_name() {
    let descriptor = CallableDescriptor::from_feval_value(
        Value::ExternalFunctionHandle("pkg..remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let CallableTarget::Resolved {
        identity,
        fallback_policy,
    } = &descriptor.target
    else {
        panic!("expected resolved target");
    };
    assert!(matches!(
        identity,
        CallableIdentity::DynamicName(SymbolName(name)) if name == "pkg..remote_inc"
    ));
    assert_eq!(
        *fallback_policy,
        CallableFallbackPolicy::RuntimeNameResolution
    );
}

#[test]
fn single_segment_external_function_handle_uses_runtime_name_resolution() {
    let descriptor = CallableDescriptor::from_feval_value(
        Value::ExternalFunctionHandle("origin".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let CallableTarget::Resolved {
        identity,
        fallback_policy,
    } = &descriptor.target
    else {
        panic!("expected resolved target");
    };
    assert!(matches!(
        identity,
        CallableIdentity::DynamicName(SymbolName(name)) if name == "origin"
    ));
    assert_eq!(
        *fallback_policy,
        CallableFallbackPolicy::RuntimeNameResolution
    );
}

#[test]
fn feval_at_prefixed_text_forwards_to_runtime_compatibility_gate() {
    let descriptor = CallableDescriptor::from_feval_value(
        Value::String("@pkg.remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    assert!(matches!(
        descriptor.target,
        CallableTarget::FevalForward(Value::String(ref text)) if text == "@pkg.remote_inc"
    ));
}

#[test]
fn feval_at_handle_external_boundary_can_use_semantic_resolver() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.remote_inc").then_some(7171)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 7171);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::String("@pkg.remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("@handle literal should resolve through semantic resolver");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn feval_string_array_at_prefixed_text_forwards_to_runtime_compatibility_gate() {
    let descriptor = CallableDescriptor::from_feval_value(
        Value::StringArray(
            StringArray::new(vec!["@pkg.remote_inc".to_string()], vec![1, 1])
                .expect("string array handle"),
        ),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    assert!(matches!(
        descriptor.target,
        CallableTarget::FevalForward(Value::StringArray(_))
    ));
}

#[test]
fn feval_string_array_at_handle_can_use_semantic_resolver() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.remote_inc").then_some(7272)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 7272);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::StringArray(
            StringArray::new(vec!["@pkg.remote_inc".to_string()], vec![1, 1])
                .expect("string array handle"),
        ),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("string-array @handle should resolve through semantic resolver");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn feval_external_function_handle_can_use_semantic_resolver() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.remote_inc").then_some(8181)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 8181);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::ExternalFunctionHandle("pkg.remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("external function handle should resolve through semantic resolver");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn feval_external_function_handle_prefers_registry_semantic_identity() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "pkg.remote_inc").then_some(9999)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 8181);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));
    let mut registry = FunctionRegistry::default();
    registry
        .names
        .insert("pkg.remote_inc".to_string(), FunctionId(8181));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::ExternalFunctionHandle("pkg.remote_inc".to_string()),
        vec![Value::Num(2.0)],
        1,
        &registry,
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("external handle should prefer registry semantic identity");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn feval_semantic_function_handle_prefers_embedded_function_id() {
    let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "inc").then_some(9999)),
    ));
    let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, args, requested_outputs| {
            assert_eq!(function, 4242);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(2.0)]);
            Box::pin(async { Ok(Value::Num(3.0)) })
        }),
    ));

    let descriptor = CallableDescriptor::from_feval_value(
        Value::BoundFunctionHandle {
            name: "inc".to_string(),
            function: 4242,
        },
        vec![Value::Num(2.0)],
        1,
        &FunctionRegistry::default(),
    );
    let value = block_on(execute_callable_descriptor(descriptor))
        .expect("semantic function handle should use embedded semantic function id");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn resolved_descriptor_infers_display_name_from_identity_when_missing() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![
            SymbolName("pkg".to_string()),
            SymbolName("remote_inc".to_string()),
        ])),
        vec![Value::Num(2.0)],
        1,
        CallableFallbackPolicy::ExternalBoundary,
        CallableCallKind::Direct,
    );

    assert_eq!(
        descriptor.metadata.display_name.as_deref(),
        Some("pkg.remote_inc")
    );
}

#[test]
fn resolved_descriptor_does_not_infer_display_name_for_malformed_external_identity() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::ExternalName(QualifiedName(vec![
            SymbolName("pkg".to_string()),
            SymbolName("".to_string()),
            SymbolName("remote_inc".to_string()),
        ])),
        vec![Value::Num(2.0)],
        1,
        CallableFallbackPolicy::ExternalBoundary,
        CallableCallKind::Direct,
    );

    assert_eq!(descriptor.metadata.display_name, None);
}

#[test]
fn anonymous_identity_error_uses_typed_identity_not_placeholder_name() {
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::AnonymousFunction(FunctionId(42)),
        vec![Value::Num(2.0)],
        1,
        CallableFallbackPolicy::None,
        CallableCallKind::Direct,
    );
    let err = block_on(execute_callable_descriptor(descriptor))
        .expect_err("anonymous identity should remain unresolved without semantic descriptor");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    assert!(
        err.message().contains("AnonymousFunction(FunctionId(42))")
            && !err.message().contains("<unnamed callable>"),
        "unexpected error: {}",
        err.message()
    );
}
