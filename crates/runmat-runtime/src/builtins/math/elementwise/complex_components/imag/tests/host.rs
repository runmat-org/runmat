use super::*;

#[test]
fn imag_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = IMAG_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = imag(X)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_scalar_real_zero() {
    let result = imag_builtin(Value::Num(-2.5)).expect("imag");
    match result {
        Value::Num(n) => assert_eq!(n, 0.0),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_complex_scalar() {
    let result = imag_builtin(Value::Complex(3.0, 4.0)).expect("imag");
    match result {
        Value::Num(n) => assert_eq!(n, 4.0),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[test]
fn imag_integer_complex_scalar_preserves_int64_value() {
    let complex = ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            runmat_value::IntegerStorage::I64(vec![i64::MIN]),
            runmat_value::IntegerStorage::I64(vec![i64::MAX]),
        )
        .unwrap(),
        vec![1, 1],
    )
    .unwrap();
    let result = imag_builtin(Value::ComplexTensor(complex)).expect("imag");
    assert_eq!(result, Value::Int(IntValue::I64(i64::MAX)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_bool_scalar_zero() {
    let result = imag_builtin(Value::Bool(true)).expect("imag");
    match result {
        Value::Num(n) => assert_eq!(n, 0.0),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_int_scalar_zero() {
    let result = imag_builtin(Value::Int(IntValue::I32(-42))).expect("imag");
    assert_eq!(result, Value::Int(IntValue::I32(0)));
}

#[test]
fn imag_typed_real_integer_tensor_zeros_from_storage_without_mirror() {
    let tensor = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![1, 9_223_372_036_854_775_809, u64::MAX]),
        vec![1, 3],
    )
    .expect("typed integer tensor");

    let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
    let Value::Tensor(output) = result else {
        panic!("expected typed integer zero tensor");
    };
    assert_eq!(output.shape, vec![1, 3]);
    assert_eq!(
        output.integer_storage(),
        Some(&runmat_value::IntegerStorage::U64(vec![0, 0, 0]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_tensor_real_is_zero() {
    let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 4.25], vec![4, 1]).unwrap();
    let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![4, 1]);
            assert!(t.materialize_f64().iter().all(|v| *v == 0.0));
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn imag_real_and_complex_single_preserve_native_class_and_shape() {
    let real = Tensor::from_f32(vec![1.0, -2.0], vec![1, 2]).unwrap();
    let Value::Tensor(output) = imag_builtin(Value::Tensor(real)).expect("imag") else {
        panic!("expected single zero tensor");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![0.0, 0.0])
    );

    let complex = ComplexTensor::from_f32(vec![(1.25, 2.5), (-3.0, 4.0)], vec![2, 1]).unwrap();
    let Value::Tensor(output) = imag_builtin(Value::ComplexTensor(complex)).expect("imag") else {
        panic!("expected single imaginary tensor");
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![2.5, 4.0])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_empty_tensor_zero_length() {
    let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
    let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![0, 3]);
            assert!(t.materialize_f64().is_empty());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn imag_empty_complex_single_preserves_native_class_and_shape() {
    let complex = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
    let Value::Tensor(output) = imag_builtin(Value::ComplexTensor(complex)).expect("imag") else {
        panic!("expected empty single imaginary tensor");
    };
    assert_eq!(output.shape, vec![0, 3]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(Vec::new())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_complex_tensor_to_tensor_of_imag_parts() {
    let complex =
        ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).expect("complex tensor");
    let result = imag_builtin(Value::ComplexTensor(complex)).expect("imag");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 1]);
            assert_eq!(t.materialize_f64(), vec![2.0, 4.5]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_logical_array_zero() {
    let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array");
    let result = imag_builtin(Value::LogicalArray(logical)).expect("imag");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), vec![0.0; 4]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_char_array_zeroes() {
    let chars = CharArray::new("Az".chars().collect(), 1, 2).expect("char array");
    let result = imag_builtin(Value::CharArray(chars)).expect("imag");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.materialize_f64(), vec![0.0, 0.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_string_error() {
    let err = imag_builtin(Value::from("hello")).expect_err("imag should error");
    let identifier = err.identifier().map(str::to_string);
    assert!(err.message().contains("expected numeric"));
    assert_eq!(identifier.as_deref(), IMAG_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_string_array_error() {
    let arr = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![2, 1]).expect("array");
    let err = imag_builtin(Value::StringArray(arr)).expect_err("imag should error");
    assert!(err.message().contains("expected numeric"));
}
