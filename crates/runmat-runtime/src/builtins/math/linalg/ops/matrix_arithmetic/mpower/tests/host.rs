use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_square_power() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result =
        mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(2))).expect("mpower");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [2, 2]);
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
fn negative_matrix_powers_share_inverse_precision_shape_and_errors() {
    for dtype in [
        runmat_value::NumericDType::F64,
        runmat_value::NumericDType::F32,
    ] {
        for (power, expected) in [
            (-1, vec![0.5, 0., -0.125, 0.25]),
            (-2, vec![0.25, 0., -0.09375, 0.0625]),
        ] {
            let matrix = Tensor::new_with_dtype(vec![2., 0., 1., 4.], vec![2, 2], dtype).unwrap();
            let result =
                mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(power))).unwrap();
            let Value::Tensor(result) = result else {
                panic!("matrix output");
            };
            assert_eq!(result.numeric_dtype(), dtype);
            assert_eq!(result.shape(), [2, 2]);
            assert_eq!(result.materialize_f64(), expected);
        }
        for (power, expected) in [
            (-1, vec![(0., -1.), (0., 0.), (0., 0.), (0., -0.5)]),
            (-2, vec![(-1., 0.), (0., 0.), (0., 0.), (-0.25, 0.)]),
        ] {
            let matrix = runmat_value::ComplexTensor::from_f64_values_with_dtype(
                vec![(0., 1.), (0., 0.), (0., 0.), (0., 2.)],
                vec![2, 2],
                dtype,
            )
            .unwrap();
            let result =
                mpower_builtin(Value::ComplexTensor(matrix), Value::Num(f64::from(power))).unwrap();
            let Value::ComplexTensor(result) = result else {
                panic!("complex matrix output");
            };
            assert_eq!(result.numeric_dtype(), dtype);
            assert_eq!(result.materialize_f64(), expected);
        }
        let empty = Tensor::new_with_dtype(Vec::new(), vec![0, 0], dtype).unwrap();
        let Value::Tensor(empty) = mpower_builtin(Value::Tensor(empty), Value::Num(-1.)).unwrap()
        else {
            panic!("empty output");
        };
        assert_eq!(empty.shape(), [0, 0]);
        assert_eq!(empty.numeric_dtype(), dtype);
    }
    let identity = Tensor::new(vec![1., 0., 0., 1.], vec![2, 2]).unwrap();
    let Value::Tensor(result) =
        mpower_builtin(Value::Tensor(identity), Value::Int(IntValue::I32(i32::MIN))).unwrap()
    else {
        panic!("wide exponent output");
    };
    assert_eq!(result.materialize_f64(), vec![1., 0., 0., 1.]);
    let singular = Tensor::new(vec![1., 2., 2., 4.], vec![2, 2]).unwrap();
    let error = mpower_builtin(Value::Tensor(singular), Value::Num(-1.)).unwrap_err();
    assert_eq!(error.identifier(), MPOWER_ERROR_INVALID_INPUT.identifier);
    assert!(std::error::Error::source(&error).is_some());
    let integers = Tensor::new_integer(
        runmat_value::IntegerStorage::I32(vec![1, 0, 0, 1]),
        vec![2, 2],
    )
    .unwrap();
    let error = mpower_builtin(Value::Tensor(integers), Value::Num(-1.)).unwrap_err();
    assert_eq!(error.identifier(), MPOWER_ERROR_INVALID_ARGUMENT.identifier);
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
