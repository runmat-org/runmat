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
        Value::FunctionHandle("plus".into()),
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
        })))
        .expect("call-port scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, _args, _requested_outputs| {
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(2.0))
            })
        },
    )))
    .expect("call-port scope");
    let result = call(
        Value::FunctionHandle("gt".into()),
        Value::Num(10.0),
        Value::Num(1.0),
    )
    .expect("bsxfun shadowed gt");

    let Value::Tensor(tensor) = result else {
        panic!("expected numeric result, got {result:?}");
    };
    assert_eq!(tensor.shape(), &[1, 1]);
    assert_eq!(tensor.materialize_f64(), vec![2.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_invokes_bound_function_for_each_broadcasted_pair() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, args, _requested_outputs| {
            let a = match &args[0] {
                Value::Num(value) => *value,
                _ => 0.0,
            };
            let b = match &args[1] {
                Value::Num(value) => *value,
                _ => 0.0,
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    a + 2.0 * b,
                ))
            })
        },
    )))
    .expect("call-port scope");
    let column = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    let result = call(
        crate::user_functions::test_support::exact_local_value("scaled_add", binding),
        Value::Tensor(column),
        Value::Tensor(row),
    )
    .expect("bsxfun closure");

    let Value::Tensor(tensor) = result else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![21.0, 22.0, 41.0, 42.0, 61.0, 62.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_rejects_non_scalar_callback_output() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 9);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, _args, _requested_outputs| {
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
                ))
            })
        },
    )))
    .expect("call-port scope");
    let err = call(
        crate::user_functions::test_support::exact_local_value("bad", binding),
        Value::Num(1.0),
        Value::Num(2.0),
    )
    .expect_err("expected callback output error");
    assert_eq!(err.identifier(), BSXFUN_ERROR_FUNCTION_ERROR.identifier);
}
