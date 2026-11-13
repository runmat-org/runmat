use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_basic_sin() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let expected: Vec<f64> = values(&tensor).into_iter().map(f64::sin).collect();
    let result = call(
        Value::FunctionHandle("sin".into()),
        vec![Value::Tensor(tensor.clone())],
    )
    .expect("arrayfun");
    match result.value() {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[2, 3]);
            assert_eq!(values(out), expected);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn arrayfun_semantic_function_handle_uses_semantic_invoker() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 78);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 78);
            assert_eq!(requested_outputs, 1);
            let [Value::Num(value)] = args else {
                panic!("expected scalar numeric argument, got {args:?}");
            };
            let value = *value;
            Box::pin(async move { rooted_single(Value::Num(value + 10.0)) })
        },
    )))
    .expect("call-port scope");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
    let handle = crate::user_functions::test_support::exact_local_value("arrayfun_target", binding);

    let result = call(handle, vec![Value::Tensor(tensor)]).expect("semantic arrayfun");
    match result.value() {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[1, 2]);
            assert_eq!(values(out), vec![11.0, 12.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn arrayfun_name_only_callback_uses_semantic_resolver() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    runtime.set_runmat_extensions_enabled(true);
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("resolved_arrayfun_target", 80)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 80);
            assert_eq!(requested_outputs, 1);
            let [Value::Num(value)] = args else {
                panic!("expected scalar numeric argument, got {args:?}");
            };
            let value = *value;
            Box::pin(async move { rooted_single(Value::Num(value + 20.0)) })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

    let result = block_on(runtime.scope(arrayfun_builtin(
        Some(runtime.clone()),
        Value::String("resolved_arrayfun_target".to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(tensor)]),
    )))
    .expect("resolved name-only arrayfun");
    match result.value() {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[1, 2]);
            assert_eq!(values(out), vec![21.0, 22.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn arrayfun_qualified_text_callback_classifies_as_external_name() {
    let callable = Callable::from_function(Value::String("pkg.callback".into()))
        .expect("qualified arrayfun callback should parse");
    assert_eq!(callable.builtin_identity(), None);
}

#[test]
fn arrayfun_external_handle_uses_semantic_resolver() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pkg.callback", 87)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 87);
            assert_eq!(requested_outputs, 1);
            let [Value::Num(value)] = args else {
                panic!("expected scalar numeric argument, got {args:?}");
            };
            let value = *value;
            Box::pin(async move { rooted_single(Value::Num(value + 30.0)) })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

    let result = block_on(runtime.scope(arrayfun_builtin(
        Some(runtime.clone()),
        Value::ExternalFunctionHandle("pkg.callback".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(tensor)]),
    )))
    .expect("resolved external-handle arrayfun");
    match result.value() {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[1, 2]);
            assert_eq!(values(out), vec![31.0, 32.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}
