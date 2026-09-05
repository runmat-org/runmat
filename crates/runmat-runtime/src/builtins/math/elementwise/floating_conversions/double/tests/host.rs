use super::*;

#[test]
fn double_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = DOUBLE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = double(X)"));
    assert!(labels.contains(&"Y = double(X, \"like\", prototype)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_scalar_num_is_identity() {
    let value = Value::Num(std::f64::consts::PI);
    let result = double_builtin(value, Vec::new()).expect("double");
    match result {
        Value::Num(n) => assert_eq!(n, std::f64::consts::PI),
        other => panic!("expected scalar Num, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_promotes_all_integer_scalar_classes() {
    for (value, expected) in [
        (IntValue::I8(-8), -8.0),
        (IntValue::I16(-16), -16.0),
        (IntValue::I32(-32), -32.0),
        (IntValue::I64(i64::MIN), i64::MIN as f64),
        (IntValue::U8(8), 8.0),
        (IntValue::U16(16), 16.0),
        (IntValue::U32(32), 32.0),
        (IntValue::U64(u64::MAX), u64::MAX as f64),
    ] {
        let result = double_builtin(Value::Int(value), Vec::new()).expect("double");
        match result {
            Value::Num(actual) => assert_eq!(actual, expected),
            other => panic!("expected scalar Num, got {other:?}"),
        }
    }
}

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
            assert_eq!(tensor.shape, vec![1, 2]);
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_logical_array_returns_tensor() {
    let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
    let result = double_builtin(Value::LogicalArray(logical), Vec::new()).expect("double");
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
fn double_char_array_converts_to_codes() {
    let chars = CharArray::new_row("AB");
    let result = double_builtin(Value::CharArray(chars), Vec::new()).expect("double");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.materialize_f64(), vec![65.0, 66.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_complex_scalar_is_identity() {
    let result = double_builtin(Value::Complex(1.5, -2.5), Vec::new()).expect("double");
    match result {
        Value::Complex(re, im) => {
            assert_eq!(re, 1.5);
            assert_eq!(im, -2.5);
        }
        other => panic!("expected complex scalar, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_tensor_preserves_shape() {
    let tensor = Tensor::new(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2]).unwrap();
    let result = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("double");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, tensor.shape);
            assert_eq!(
                t.into_numeric_storage().expect("double storage"),
                NumericStorage::F64(vec![1.25, 2.5, 3.75, 4.5])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_tensor_reads_typed_integer_storage_and_clears_class() {
    let tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2]).unwrap();

    let result = double_builtin(Value::Tensor(tensor), Vec::new()).expect("double");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(
                t.into_numeric_storage().expect("double storage"),
                NumericStorage::F64(vec![(1_u64 << 63) as f64, u64::MAX as f64])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_complex_tensor_reads_typed_integer_storage_and_clears_class() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![-2, 3]),
        IntegerStorage::I16(vec![5, -7]),
    )
    .unwrap();
    let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

    let result = double_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("double");
    match result {
        Value::ComplexTensor(t) => {
            assert!(t.integer_storage().is_none());
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.materialize_f64(), vec![(-2.0, 5.0), (3.0, -7.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_sparse_tensor_preserves_sparsity_and_converts_storage() {
    let sparse = SparseTensor::new_f32(3, 2, vec![0, 1, 2], vec![1, 2], vec![4.0, -1.0]).unwrap();
    let result = double_builtin(Value::SparseTensor(sparse), Vec::new()).expect("double");
    let Value::SparseTensor(output) = result else {
        panic!("expected sparse tensor");
    };
    assert_eq!(output.shape(), vec![3, 2]);
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F64));
    assert_eq!(output.as_f64_slice(), Some(&[4.0, -1.0][..]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_sparse_tensor_reads_typed_integer_storage_and_clears_class() {
    let sparse = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![1, 0],
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
    )
    .unwrap();
    let result = double_builtin(Value::SparseTensor(sparse), Vec::new()).expect("double");
    let Value::SparseTensor(output) = result else {
        panic!("expected sparse tensor");
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F64));
    assert!(output.integer_storage().is_none());
    assert_eq!(
        output.as_f64_slice(),
        Some(&[i64::MIN as f64, i64::MAX as f64][..])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_converts_string_scalars_and_arrays() {
    assert_eq!(
        double_builtin(Value::String(" 12.5 ".into()), Vec::new()).unwrap(),
        Value::Num(12.5)
    );
    let invalid = double_builtin(Value::String("not a number".into()), Vec::new()).unwrap();
    assert!(matches!(invalid, Value::Num(value) if value.is_nan()));

    let strings = StringArray::new(
        vec!["1".into(), "-2.25".into(), "missing".into(), "Inf".into()],
        vec![2, 2],
    )
    .unwrap();
    let Value::Tensor(output) = double_builtin(Value::StringArray(strings), Vec::new()).unwrap()
    else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape, vec![2, 2]);
    let values = output.materialize_f64();
    assert_eq!(values[0..2], [1.0, -2.25]);
    assert!(values[2].is_nan());
    assert_eq!(values[3], f64::INFINITY);
}
