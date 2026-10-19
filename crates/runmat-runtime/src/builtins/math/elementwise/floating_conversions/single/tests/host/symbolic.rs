use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_converts_symbolic_constants() {
    let result = single_builtin(
        Value::Symbolic(SymbolicExpr::constant(std::f64::consts::PI)),
        Vec::new(),
    )
    .expect("single");

    let Value::Tensor(tensor) = result else {
        panic!("expected native single symbolic constant");
    };
    assert_eq!(
        tensor.as_f32_slice(),
        Some(&[std::f64::consts::PI as f32][..])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_converts_symbolic_array_constants() {
    let array = SymbolicArray::new(
        vec![SymbolicExpr::constant(1.25), SymbolicExpr::constant(2.5)],
        vec![1, 2],
    )
    .unwrap();

    let result = single_builtin(Value::SymbolicArray(array), Vec::new()).expect("single");

    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 2]);
            assert_eq!(
                tensor.materialize_f64(),
                vec![(1.25f32) as f64, (2.5f32) as f64]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_rejects_symbolic_array_variables() {
    let array = SymbolicArray::new(vec![SymbolicExpr::variable("x")], vec![1, 1]).unwrap();

    let err = single_builtin(Value::SymbolicArray(array), Vec::new())
        .expect_err("symbolic variable should not convert");

    assert_eq!(err.identifier(), SINGLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("conversion to single from sym"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_rejects_symbolic_variables() {
    let err = single_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
        .expect_err("symbolic variable should not convert");

    assert_eq!(err.identifier(), SINGLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("conversion to single from sym"));
}
