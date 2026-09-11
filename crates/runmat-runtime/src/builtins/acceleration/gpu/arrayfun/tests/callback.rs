use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_basic_sin() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let expected: Vec<f64> = values(&tensor).into_iter().map(f64::sin).collect();
    let result = call(
        Value::FunctionHandle("sin".to_string()),
        vec![Value::Tensor(tensor.clone())],
    )
    .expect("arrayfun");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![2, 3]);
            assert_eq!(values(&out), expected);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn arrayfun_semantic_function_handle_uses_semantic_invoker() {
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 78);
            assert_eq!(requested_outputs, 1);
            let [Value::Num(value)] = args else {
                panic!("expected scalar numeric argument, got {args:?}");
            };
            let value = *value;
            Box::pin(
                async move { crate::sequence::single_value_sequence(Value::Num(value + 10.0)) },
            )
        },
    )));
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
    let handle = Value::BoundFunctionHandle {
        name: "arrayfun_target".into(),
        function: 78,
    };

    let result = call(handle, vec![Value::Tensor(tensor)]).expect("semantic arrayfun");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(values(&out), vec![11.0, 12.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn arrayfun_name_only_callback_uses_semantic_resolver() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "resolved_arrayfun_target").then_some(80)
        })));
    let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 80);
            assert_eq!(requested_outputs, 1);
            let [Value::Num(value)] = args else {
                panic!("expected scalar numeric argument, got {args:?}");
            };
            let value = *value;
            Box::pin(
                async move { crate::sequence::single_value_sequence(Value::Num(value + 20.0)) },
            )
        },
    )));
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

    let result = call(
        Value::String("resolved_arrayfun_target".to_string()),
        vec![Value::Tensor(tensor)],
    )
    .expect("resolved name-only arrayfun");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(values(&out), vec![21.0, 22.0]);
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
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "pkg.callback").then_some(87)
        })));
    let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 87);
            assert_eq!(requested_outputs, 1);
            let [Value::Num(value)] = args else {
                panic!("expected scalar numeric argument, got {args:?}");
            };
            let value = *value;
            Box::pin(
                async move { crate::sequence::single_value_sequence(Value::Num(value + 30.0)) },
            )
        },
    )));
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

    let result = call(
        Value::ExternalFunctionHandle("pkg.callback".to_string()),
        vec![Value::Tensor(tensor)],
    )
    .expect("resolved external-handle arrayfun");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(values(&out), vec![31.0, 32.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}
