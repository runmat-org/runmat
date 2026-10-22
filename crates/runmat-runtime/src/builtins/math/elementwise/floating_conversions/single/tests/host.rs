use super::*;

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
        let Value::Tensor(output) = single_builtin(Value::CharArray(chars), vec![]).unwrap() else {
            panic!("dense character conversion");
        };
        assert_eq!(output.shape(), shape.as_slice());
        assert_eq!(output.materialize_f64(), expected);
    }
}

#[test]
fn single_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = SINGLE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = single(X)"));
    assert!(labels.contains(&"Y = single(X, \"like\", prototype)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_scalar_rounds_to_f32() {
    let value = Value::Num(std::f64::consts::PI);
    let result = single_builtin(value, Vec::new()).expect("single");
    let Value::Tensor(tensor) = result else {
        panic!("expected native single scalar tensor");
    };
    assert_eq!(tensor.shape(), &[1, 1]);
    assert_eq!(
        tensor.as_f32_slice(),
        Some(&[std::f64::consts::PI as f32][..])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_tensor_preserves_shape() {
    let tensor = Tensor::new(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2]).unwrap();
    let result = single_builtin(Value::Tensor(tensor), Vec::new()).expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(
                t.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![1.25, 2.5, 3.75, 4.5])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_tensor_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();

    let result = single_builtin(Value::Tensor(tensor), Vec::new()).expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(
                t.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_sparse_tensor_preserves_sparsity_and_converts_storage() {
    let sparse = SparseTensor::new(
        3,
        2,
        vec![0, 1, 2],
        vec![1, 2],
        vec![std::f64::consts::PI, -1.25],
    )
    .unwrap();
    let result = single_builtin(Value::SparseTensor(sparse), Vec::new()).expect("single");
    let Value::SparseTensor(output) = result else {
        panic!("expected sparse tensor");
    };
    assert_eq!(output.shape(), vec![3, 2]);
    assert_eq!(
        output.numeric_dtype(),
        Some(runmat_value::NumericDType::F32)
    );
    assert_eq!(
        output.as_f32_slice(),
        Some(&[std::f64::consts::PI as f32, -1.25][..])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_complex_tensor_rounds_both_components() {
    let tensor = ComplexTensor::new(
        vec![(1.234567, -9.876543), (0.3333333, 0.6666667)],
        vec![1, 2],
    )
    .unwrap();
    let result = single_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("single");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape(), &[1, 2]);
            let expected: Vec<(f64, f64)> = vec![
                ((1.234567f32) as f64, (-9.876543f32) as f64),
                ((0.3333333f32) as f64, (0.6666667f32) as f64),
            ];
            assert_eq!(t.materialize_f64(), expected);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_complex_tensor_reads_typed_integer_storage_exactly() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![1, -2]),
        IntegerStorage::I16(vec![3, -4]),
    )
    .expect("complex integer storage");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

    let result = single_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("single");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape(), &[1, 2]);
            assert_eq!(t.integer_storage(), None);
            assert_eq!(t.materialize_f64(), vec![(1.0, 3.0), (-2.0, -4.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_char_array_produces_codes() {
    let chars = CharArray::from_code_units(
        [0x0041, 0xD83D, 0xDE00, 0xD800, 0x0042, 0xDC00, 0, 0xFFFF]
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .to_vec(),
        2,
        4,
    )
    .unwrap();
    let result = single_builtin(Value::CharArray(chars), Vec::new()).expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 4]);
            assert_eq!(
                t.as_f32_slice(),
                Some(&[65.0, 66.0, 55357.0, 56320.0, 56832.0, 0.0, 55296.0, 65535.0][..])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_errors_on_string_input() {
    let err = single_builtin(Value::String("hello".to_string().into()), Vec::new())
        .expect_err("expected error");
    assert_eq!(err.identifier(), SINGLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("string"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn single_preserves_low_rank_real_and_complex_geometry_while_rounding_class() {
    let scalar = Tensor::new(vec![1.25], Vec::new()).expect("rank-zero double");
    let Value::Tensor(scalar) = single_builtin(Value::Tensor(scalar), Vec::new()).expect("single")
    else {
        panic!("single tensor output")
    };
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(scalar.as_f32_slice(), Some(&[1.25_f32][..]));

    let vector =
        ComplexTensor::new(vec![(1.0, -2.0), (3.0, 4.0)], vec![2]).expect("complex vector");
    let Value::ComplexTensor(vector) =
        single_builtin(Value::ComplexTensor(vector), Vec::new()).expect("single complex")
    else {
        panic!("single complex tensor output")
    };
    assert_eq!(vector.shape(), &[2]);
    assert_eq!(vector.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(
        vector
            .as_f32_slice()
            .unwrap()
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>(),
        vec![(1.0, -2.0), (3.0, 4.0)]
    );
}
