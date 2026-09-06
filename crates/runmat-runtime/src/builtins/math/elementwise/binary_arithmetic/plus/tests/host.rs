use super::*;

#[test]
fn plus_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = PLUS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = plus(A, B)"));
    assert!(labels.contains(&"C = plus(A, B, \"like\", prototype)"));
}

#[test]
fn plus_parser_error_has_stable_identifier() {
    let err = plus_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
        .expect_err("expected parser error");
    assert_eq!(err.identifier(), PLUS_ERROR_INVALID_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_scalar_numbers() {
    let result = plus_builtin(Value::Num(2.0), Value::Num(3.5), Vec::new()).expect("plus");
    match result {
        Value::Num(v) => assert!((v - 5.5).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_matrix_scalar() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = plus_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("plus");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(
                t.as_f64_slice().expect("double output"),
                &[3.0, 4.0, 5.0, 6.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn plus_typed_sparse_uint64_uses_exact_sparse_route() {
    let lhs = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    )
    .unwrap();
    let rhs = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![1, 0],
        IntegerStorage::U64(vec![7, 1]),
    )
    .unwrap();
    let Value::SparseTensor(result) = plus_builtin(
        Value::SparseTensor(lhs),
        Value::SparseTensor(rhs),
        Vec::new(),
    )
    .expect("plus") else {
        panic!("expected typed sparse result");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            9_007_199_254_740_993,
            7,
            1,
            u64::MAX
        ]))
    );
}

#[test]
fn plus_dense_integer_arrays_preserve_exact_storage_without_mirror() {
    let lhs = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX, (1_u64 << 63) + 1]),
        vec![2, 1],
    )
    .expect("lhs");
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![1, 7, 2]), vec![1, 3]).expect("rhs");

    let result =
        plus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("integer plus");
    let Value::Tensor(result) = result else {
        panic!("expected integer tensor");
    };
    assert_eq!(result.shape, vec![2, 3]);
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            u64::MAX,
            (1_u64 << 63) + 2,
            u64::MAX,
            (1_u64 << 63) + 8,
            u64::MAX,
            (1_u64 << 63) + 3
        ]))
    );

    let scalar_tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![i16::MAX]), vec![1, 1]).expect("scalar");
    assert_eq!(
        plus_builtin(Value::Tensor(scalar_tensor), Value::Num(1.0), Vec::new())
            .expect("scalar plus"),
        Value::Int(IntValue::I16(i16::MAX))
    );
}

#[test]
fn plus_float_arrays_preserve_native_single_class() {
    let lhs = Tensor::from_f32(vec![1.25, -4.0], vec![1, 2]).unwrap();
    let rhs = Tensor::from_f32(vec![2.0, 0.5], vec![1, 2]).unwrap();
    let Value::Tensor(result) =
        plus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
    else {
        panic!("expected single tensor");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3.25, -3.5])
    );

    let lhs = Tensor::new(vec![0.1, 0.2], vec![1, 2]).unwrap();
    let rhs = Tensor::from_f32(vec![0.2, 0.3], vec![1, 2]).unwrap();
    let Value::Tensor(result) =
        plus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
    else {
        panic!("expected mixed floating tensor");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![
            (0.1_f64 + f64::from(0.2_f32)) as f32,
            (0.2_f64 + f64::from(0.3_f32)) as f32,
        ])
    );
}

#[test]
fn plus_complex_arrays_preserve_native_single_class() {
    let lhs = ComplexTensor::from_f32(vec![(1.25, -2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
    let rhs = Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap();
    let Value::ComplexTensor(result) =
        plus_builtin(Value::ComplexTensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
    else {
        panic!("expected complex single tensor");
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(1.75_f32, -2.0_f32), (4.0_f32, 4.0_f32)])
    );
}

#[test]
fn plus_mixed_complex_floating_inputs_return_single_without_scalar_collapse() {
    let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let double = ComplexTensor::new(vec![(3.0, -1.0)], vec![1, 1]).unwrap();
    for (lhs, rhs) in [
        (single.clone(), double.clone()),
        (double.clone(), single.clone()),
    ] {
        let result = plus_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex plus");
        let Value::ComplexTensor(result) = result else {
            panic!("expected one-element complex single tensor");
        };
        assert_eq!(
            result.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![(4.0, 1.0)])
        );
    }
}

#[test]
fn plus_real_complex_single_reverse_path_and_empty_class_are_preserved() {
    let real = Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap();
    let complex = ComplexTensor::from_f32(vec![(1.25, -2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
    let result = plus_builtin(
        Value::Tensor(real),
        Value::ComplexTensor(complex),
        Vec::new(),
    )
    .expect("real-complex plus");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(1.75, -2.0), (4.0, 4.0)])
    );
    let lhs = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let rhs = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
    let result = plus_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("empty complex plus");
    let Value::ComplexTensor(result) = result else {
        panic!("expected empty complex single tensor");
    };
    assert_eq!(result.shape, vec![0, 2]);
    assert_eq!(result.as_f32_slice(), Some(&[][..]));
}

#[test]
fn plus_like_complex_conversion_preserves_single_storage() {
    let tensor = Tensor::from_f32(vec![2.0, 3.0], vec![2, 1]).unwrap();
    let result =
        block_on(super::real_to_complex(Value::Tensor(tensor))).expect("complex conversion");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(2.0, 0.0), (3.0, 0.0)])
    );
}

#[test]
fn plus_rejects_real_integer_with_floating_complex() {
    let integer = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 63) + 1, u64::MAX]),
            vec![1, 2],
        )
        .unwrap(),
    );
    let complex =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap());
    for (lhs, rhs) in [
        (integer.clone(), complex.clone()),
        (complex.clone(), integer.clone()),
    ] {
        let error = plus_builtin(lhs, rhs, Vec::new()).unwrap_err();
        assert!(error
            .message()
            .contains("complex integer arithmetic is not supported"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_row_column_broadcast() {
    let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    let result = plus_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
        .expect("broadcast plus");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 3]);
            let expected = vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0];
            assert_eq!(t.as_f64_slice().expect("double output"), expected);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_complex_inputs() {
    let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = plus_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex plus");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected = [(3.0, 1.0), (2.0, -3.0)];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
            }
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_char_input() {
    let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
    let result =
        plus_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("char plus");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 3]);
            assert_eq!(
                t.as_f64_slice().expect("double output"),
                &[67.0, 68.0, 69.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_logical_input_promotes_to_double() {
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    let tensor = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![2, 2]).unwrap();
    let result = plus_builtin(
        Value::LogicalArray(logical),
        Value::Tensor(tensor),
        Vec::new(),
    )
    .expect("logical");
    match result {
        Value::Tensor(t) => {
            assert_eq!(
                t.as_f64_slice().expect("double output"),
                &[3.0, 2.0, 4.0, 3.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_dimension_mismatch_errors() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = plus_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).unwrap_err();
    assert!(
        err.message().contains("plus"),
        "unexpected error message: {err}"
    );
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_same_class_integer_inputs_preserve_class() {
    let lhs = Value::Int(IntValue::I32(3));
    let rhs = Value::Int(IntValue::I32(5));
    let result = plus_builtin(lhs, rhs, Vec::new()).expect("plus");
    assert_eq!(result, Value::Int(IntValue::I32(8)));
}
