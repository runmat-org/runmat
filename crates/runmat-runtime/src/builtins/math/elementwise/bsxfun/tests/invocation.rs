use super::*;

#[test]
fn bsxfun_callable_text_is_gated_but_function_handles_are_ordinary() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(Value::from("@plus"), Value::Num(1.0), Value::Num(2.0))
        .expect_err("strict mode must reject callable text");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BsxfunTextCallableExtension")
    );
    call(
        Value::FunctionHandle("plus".to_string()),
        Value::Num(1.0),
        Value::Num(2.0),
    )
    .expect("function handle is documented");
    drop(_strict);

    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = call(Value::from("@plus"), Value::Num(1.0), Value::Num(2.0))
        .expect("RunMat mode admits callable text");
    let Value::Tensor(result) = result else {
        panic!("expected double tensor");
    };
    assert_eq!(result.materialize_f64(), vec![3.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_does_not_logicalize_shadowed_callback_names() {
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "gt").then_some(17)
        })));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, _args, _requested_outputs| Box::pin(async move { Ok(Value::Num(2.0)) }),
    )));
    let result = call(
        Value::FunctionHandle("gt".to_string()),
        Value::Num(10.0),
        Value::Num(1.0),
    )
    .expect("bsxfun shadowed gt");

    let Value::Tensor(tensor) = result else {
        panic!("expected numeric result, got {result:?}");
    };
    assert_eq!(tensor.shape, vec![1, 1]);
    assert_eq!(tensor.materialize_f64(), vec![2.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_invokes_bound_function_for_each_broadcasted_pair() {
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, args, _requested_outputs| {
            let args = args.to_vec();
            Box::pin(async move {
                let a = match &args[0] {
                    Value::Num(value) => *value,
                    _ => 0.0,
                };
                let b = match &args[1] {
                    Value::Num(value) => *value,
                    _ => 0.0,
                };
                Ok(Value::Num(a + 2.0 * b))
            })
        },
    )));
    let column = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    let result = call(
        Value::BoundFunctionHandle {
            name: "scaled_add".into(),
            function: 7,
        },
        Value::Tensor(column),
        Value::Tensor(row),
    )
    .expect("bsxfun closure");

    let Value::Tensor(tensor) = result else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![21.0, 22.0, 41.0, 42.0, 61.0, 62.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_rejects_non_scalar_callback_output() {
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, _args, _requested_outputs| {
            Box::pin(async move {
                Ok(Value::Tensor(
                    Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
                ))
            })
        },
    )));
    let err = call(
        Value::BoundFunctionHandle {
            name: "bad".into(),
            function: 9,
        },
        Value::Num(1.0),
        Value::Num(2.0),
    )
    .expect_err("expected callback output error");
    assert_eq!(err.identifier(), BSXFUN_ERROR_FUNCTION_ERROR.identifier);
}
