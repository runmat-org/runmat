use super::*;

#[test]
fn real_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = REAL_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = real(X)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_scalar_num() {
    let result = real_builtin(Value::Num(-2.5)).expect("real");
    match result {
        Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_complex_scalar() {
    let result = real_builtin(Value::Complex(3.0, 4.0)).expect("real");
    match result {
        Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_int_scalar_preserves_integer_class() {
    let result = real_builtin(Value::Int(IntValue::I32(7))).expect("real");
    assert_eq!(result, Value::Int(IntValue::I32(7)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_complex_tensor_to_real_tensor() {
    let complex =
        ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.0)], vec![2, 1]).expect("complex tensor");
    let result = real_builtin(Value::ComplexTensor(complex)).expect("real");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 1]);
            assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
            assert!((t.materialize_f64()[1] + 3.0).abs() < 1e-12);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn real_complex_single_preserves_native_class_shape_and_empty_storage() {
    let complex = ComplexTensor::from_f32(vec![(1.25, 2.5), (-3.0, 4.0)], vec![2, 1]).unwrap();
    let Value::Tensor(output) = real_builtin(Value::ComplexTensor(complex)).expect("real") else {
        panic!("expected single real tensor");
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![1.25, -3.0])
    );

    let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 4]).unwrap();
    let Value::Tensor(output) = real_builtin(Value::ComplexTensor(empty)).expect("real") else {
        panic!("expected empty single real tensor");
    };
    assert_eq!(output.shape, vec![0, 4]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(Vec::new())
    );
}

#[test]
fn real_integer_complex_tensor_preserves_uint64_values() {
    let complex = ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            runmat_value::IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
            runmat_value::IntegerStorage::U64(vec![2, 3]),
        )
        .unwrap(),
        vec![1, 2],
    )
    .unwrap();
    let result = real_builtin(Value::ComplexTensor(complex)).expect("real");
    let Value::Tensor(tensor) = result else {
        panic!("expected typed real tensor");
    };
    assert_eq!(
        tensor.integer_storage(),
        Some(&runmat_value::IntegerStorage::U64(vec![
            9_223_372_036_854_775_809,
            u64::MAX,
        ]))
    );
}

#[test]
fn real_integer_complex_tensor_reads_component_storage_exactly() {
    let complex = ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            runmat_value::IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            runmat_value::IntegerStorage::I64(vec![7, -8]),
        )
        .unwrap(),
        vec![2, 1],
    )
    .unwrap();

    let result = real_builtin(Value::ComplexTensor(complex)).expect("real");
    let Value::Tensor(tensor) = result else {
        panic!("expected typed real tensor");
    };
    assert_eq!(tensor.shape, vec![2, 1]);
    assert_eq!(
        tensor.integer_storage(),
        Some(&runmat_value::IntegerStorage::I64(
            vec![i64::MIN, i64::MAX,]
        ))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_logical_array_to_numeric() {
    let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array");
    let result = real_builtin(Value::LogicalArray(logical)).expect("real");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 1.0, 0.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_char_array_codes() {
    let chars = CharArray::new("AZ".chars().collect(), 1, 2).expect("char array");
    let result = real_builtin(Value::CharArray(chars)).expect("real");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.materialize_f64(), vec![65.0, 90.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_string_error() {
    let err = real_builtin(Value::from("hello")).expect_err("real should error");
    let identifier = err.identifier().map(str::to_string);
    assert!(err.message().contains("expected numeric"));
    assert_eq!(identifier.as_deref(), REAL_ERROR_INVALID_INPUT.identifier);
}
