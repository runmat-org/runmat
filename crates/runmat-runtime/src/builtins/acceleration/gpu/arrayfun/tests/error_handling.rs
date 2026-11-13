use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_error_handler_recovers() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 991);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, requested_outputs| {
            assert_eq!(function, 991);
            assert_eq!(requested_outputs, 1);
            let seed = arguments.first().cloned().expect("captured seed");
            Box::pin(async move { rooted_single(seed) })
        },
    )))
    .expect("call-port scope");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let handler = Value::Closure(Closure::from_data(runmat_value::ClosureData {
        lexical_environment: None,
        function_name: "arrayfun_error_handler_fixture".into(),
        binding: Some(binding),
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: vec![Value::Num(42.0)].into(),
    }));
    let result = call(
        Value::FunctionHandle("nonexistent_builtin".into()),
        vec![
            Value::Tensor(tensor),
            Value::String("ErrorHandler".into()),
            handler,
        ],
    )
    .expect("arrayfun error handler");
    match result.value() {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[3, 1]);
            assert_eq!(values(out), vec![42.0, 42.0, 42.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_error_without_handler_propagates_identifier() {
    let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let err = call(
        Value::FunctionHandle("nonexistent_builtin".into()),
        vec![Value::Tensor(tensor)],
    )
    .expect_err("expected unresolved function error");
    assert_eq!(
        err.identifier(),
        ARRAYFUN_ERROR_UNDEFINED_FUNCTION.identifier,
        "unexpected error: {}",
        err.message()
    );
}
