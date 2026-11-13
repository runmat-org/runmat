use super::*;

mod strings;
mod symbolic;

#[test]
fn character_conversion_preserves_column_major_order_and_full_nd_shape() {
    for (source, shape, expected) in [
        (
            "abcdef",
            vec![2, 3],
            vec![97.0, 100.0, 98.0, 101.0, 99.0, 102.0],
        ),
        (
            "abcdefgh",
            vec![2, 2, 2],
            vec![97.0, 99.0, 98.0, 100.0, 101.0, 103.0, 102.0, 104.0],
        ),
        ("", vec![1, 0, 3], vec![]),
    ] {
        let chars = runmat_value::CharArray::from_unicode_scalars_with_shape(
            source.chars().collect(),
            shape.clone(),
        )
        .unwrap();
        let Value::Tensor(output) = double_builtin(Value::CharArray(chars), vec![]).unwrap() else {
            panic!("dense character conversion");
        };
        assert_eq!(output.shape(), shape.as_slice());
        assert_eq!(output.materialize_f64(), expected);
    }
}

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
fn double_logical_array_returns_tensor() {
    let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
    let result = double_builtin(Value::LogicalArray(logical), Vec::new()).expect("double");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 1.0, 0.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_char_array_converts_to_codes() {
    let chars = CharArray::from_code_units(
        [0x0041, 0xD83D, 0xDE00, 0xD800, 0x0042, 0xDC00, 0, 0xFFFF]
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .to_vec(),
        2,
        4,
    )
    .unwrap();
    let result = double_builtin(Value::CharArray(chars), Vec::new()).expect("double");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 4]);
            assert_eq!(
                t.as_f64_slice(),
                Some(&[65.0, 66.0, 55357.0, 56320.0, 56832.0, 0.0, 55296.0, 65535.0][..])
            );
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
            assert_eq!(t.shape(), tensor.shape());
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
            assert_eq!(t.shape(), &[1, 2]);
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
            assert_eq!(t.shape(), &[1, 2]);
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
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn double_preserves_low_rank_real_and_complex_geometry_while_promoting_class() {
    let scalar = Tensor::from_f32(vec![1.25], Vec::new()).expect("rank-zero single");
    let Value::Tensor(scalar) = double_builtin(Value::Tensor(scalar), Vec::new()).expect("double")
    else {
        panic!("tensor input remains a tensor")
    };
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.numeric_dtype(), NumericDType::F64);
    assert_eq!(scalar.materialize_f64(), vec![1.25]);

    let vector =
        ComplexTensor::from_f32(vec![(1.0, -2.0), (3.0, 4.0)], vec![2]).expect("complex vector");
    let Value::ComplexTensor(vector) =
        double_builtin(Value::ComplexTensor(vector), Vec::new()).expect("double complex")
    else {
        panic!("complex tensor input remains a complex tensor")
    };
    assert_eq!(vector.shape(), &[2]);
    assert_eq!(vector.numeric_dtype(), NumericDType::F64);
    assert_eq!(vector.materialize_f64(), vec![(1.0, -2.0), (3.0, 4.0)]);
}
