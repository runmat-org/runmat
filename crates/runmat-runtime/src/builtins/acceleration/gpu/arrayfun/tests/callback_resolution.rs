use super::*;

#[test]
fn arrayfun_single_segment_external_handle_uses_runtime_name_resolution() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("callback", 887)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 887);
            assert_eq!(requested_outputs, 1);
            let [Value::Num(value)] = args else {
                panic!("expected scalar numeric argument, got {args:?}");
            };
            let value = *value;
            Box::pin(async move { rooted_single(Value::Num(value + 40.0)) })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

    let result = block_on(runtime.scope(arrayfun_builtin(
        Some(runtime.clone()),
        Value::ExternalFunctionHandle("callback".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(tensor)]),
    )))
    .expect("single-segment external-handle arrayfun should resolve via runtime-name policy");
    match result.value() {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[1, 2]);
            assert_eq!(values(out), vec![41.0, 42.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn arrayfun_external_handle_prefers_semantic_handle_binding_when_resolved() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pkg.callback", 87)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            assert_eq!(function, 87);
            assert_eq!(arguments, &[Value::Num(4.0)]);
            Box::pin(async { rooted_single(Value::Num(8.0)) })
        },
    )))
    .expect("call-port scope");
    let callable = Callable::from_function(Value::ExternalFunctionHandle("pkg.callback".into()))
        .expect("external handle should parse");
    drop(entry);
    assert_eq!(
        block_on(runtime.scope(callable.call(Some(&runtime), &[Value::Num(4.0)]))).unwrap(),
        Value::Num(8.0)
    );
}

#[test]
fn arrayfun_name_only_closure_prefers_semantic_handle_binding_when_resolved() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pkg.callback", 187)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            assert_eq!(function, 187);
            assert_eq!(arguments, &[Value::Num(5.0), Value::Num(4.0)]);
            Box::pin(async { rooted_single(Value::Num(9.0)) })
        },
    )))
    .expect("call-port scope");
    let callable = Callable::from_function(Value::Closure(Closure::from_data(
        runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "pkg.callback".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(5.0)].into(),
        },
    )))
    .expect("closure callback should parse");
    drop(entry);
    assert_eq!(
        block_on(runtime.scope(callable.call(Some(&runtime), &[Value::Num(4.0)]))).unwrap(),
        Value::Num(9.0)
    );
}

#[test]
fn arrayfun_name_only_closure_call_uses_semantic_resolver_when_unbound() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pkg.callback", 287)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 287);
            assert_eq!(requested_outputs, 1);
            assert_eq!(args, &[Value::Num(5.0), Value::Num(4.0)]);
            Box::pin(async { rooted_single(Value::Num(9.0)) })
        },
    )))
    .expect("call-port scope");
    let callable = Callable::from_function(Value::Closure(Closure::from_data(
        runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "pkg.callback".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(5.0)].into(),
        },
    )))
    .expect("closure callback should parse");
    drop(entry);
    let value = block_on(runtime.scope(callable.call(Some(&runtime), &[Value::Num(4.0)])))
        .expect("closure call");
    assert_eq!(value, Value::Num(9.0));
}

#[test]
fn arrayfun_external_handle_errors_as_undefined_when_unresolved() {
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)))
            .expect("call-port scope");
    let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");

    let err = call(
        Value::ExternalFunctionHandle("pkg.callback".into()),
        vec![Value::Tensor(tensor)],
    )
    .expect_err("unresolved external callback should error");
    assert_eq!(
        err.identifier(),
        ARRAYFUN_ERROR_UNDEFINED_FUNCTION.identifier,
        "unexpected error: {}",
        err.message()
    );
    assert!(
        err.message().contains("ExternalName(QualifiedName"),
        "unexpected error: {err:?}"
    );
    assert!(
        err.message().contains("Undefined function 'pkg.callback'"),
        "external callback should retain its readable name alongside its typed identity: {err:?}"
    );
}

#[test]
fn arrayfun_malformed_external_handle_errors_as_undefined_when_unresolved() {
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)))
            .expect("call-port scope");
    let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");

    let err = call(
        Value::ExternalFunctionHandle("pkg..callback".into()),
        vec![Value::Tensor(tensor)],
    )
    .expect_err("malformed unresolved external callback should error");
    assert_eq!(
        err.identifier(),
        ARRAYFUN_ERROR_UNDEFINED_FUNCTION.identifier,
        "unexpected error: {}",
        err.message()
    );
    assert!(
        err.message().contains("pkg..callback"),
        "unexpected error: {err:?}"
    );
}
