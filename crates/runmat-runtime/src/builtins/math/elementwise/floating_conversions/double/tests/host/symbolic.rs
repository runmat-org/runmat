use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_converts_symbolic_constants() {
    let result =
        double_builtin(Value::Symbolic(SymbolicExpr::constant(42.5)), Vec::new()).expect("double");

    assert_eq!(result, Value::Num(42.5));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_converts_symbolic_array_constants() {
    let array = SymbolicArray::new(
        vec![SymbolicExpr::constant(1.5), SymbolicExpr::constant(2.5)],
        vec![1, 2],
    )
    .unwrap();

    let result = double_builtin(Value::SymbolicArray(array), Vec::new()).expect("double");

    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 2]);
            assert_eq!(tensor.materialize_f64(), vec![1.5, 2.5]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_rejects_symbolic_array_variables() {
    let array = SymbolicArray::new(vec![SymbolicExpr::variable("x")], vec![1, 1]).unwrap();

    let err = double_builtin(Value::SymbolicArray(array), Vec::new())
        .expect_err("symbolic variable should not convert");

    assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("conversion to double from sym"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_rejects_symbolic_variables() {
    let err = double_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
        .expect_err("symbolic variable should not convert");

    assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("conversion to double from sym"));
}
