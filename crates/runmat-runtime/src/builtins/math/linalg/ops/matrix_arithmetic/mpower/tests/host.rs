use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_square_power() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result =
        mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(2))).expect("mpower");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), vec![7.0, 10.0, 15.0, 22.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn zero_exponent_returns_identity() {
    let matrix = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
    let result =
        mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(0))).expect("mpower");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn scalar_inputs_match_standard_power() {
    let result = mpower_builtin(Value::Num(4.0), Value::Num(0.5)).expect("mpower");
    match result {
        Value::Num(v) => assert!((v - 2.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn non_integer_exponent_errors() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let err = unwrap_error(mpower_builtin(Value::Tensor(matrix), Value::Num(1.5)).unwrap_err());
    assert_eq!(err.identifier(), MPOWER_ERROR_INVALID_ARGUMENT.identifier);
    assert!(
        err.message()
            .contains("Matrix power requires integer exponent"),
        "{err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn negative_exponent_errors() {
    let matrix = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let err = unwrap_error(
        mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(-1))).unwrap_err(),
    );
    assert_eq!(err.identifier(), MPOWER_ERROR_INVALID_ARGUMENT.identifier);
    assert!(
        err.message()
            .contains("Negative matrix powers not supported yet"),
        "{err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn non_square_matrix_errors() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let err = unwrap_error(
        mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(2))).unwrap_err(),
    );
    assert_eq!(err.identifier(), MPOWER_ERROR_INVALID_INPUT.identifier);
    assert!(
        err.message()
            .contains("Matrix must be square for matrix power"),
        "{err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_scalar_power() {
    let result =
        mpower_builtin(Value::Complex(2.0, 1.0), Value::Int(IntValue::I32(3))).expect("mpower");
    match result {
        Value::Complex(re, im) => {
            assert!((re - 2.0).abs() < 1e-12);
            assert!((im - 11.0).abs() < 1e-12);
        }
        other => panic!("expected complex scalar, got {other:?}"),
    }
}
