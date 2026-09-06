use super::*;

#[test]
fn minus_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = MINUS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = minus(A, B)"));
    assert!(labels.contains(&"C = minus(A, B, \"like\", prototype)"));
}

#[test]
fn minus_parser_error_has_stable_identifier() {
    let err = minus_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
        .expect_err("expected parser error");
    assert_eq!(err.identifier(), MINUS_ERROR_INVALID_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_scalar_numbers() {
    let result = minus_builtin(Value::Num(2.0), Value::Num(3.5), Vec::new()).expect("minus");
    match result {
        Value::Num(v) => assert!((v + 1.5).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_matrix_scalar() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = minus_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("minus");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(
                t.as_f64_slice().expect("double result"),
                &[-1.0, 0.0, 1.0, 2.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn minus_like_complex_conversion_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-2, 3]), vec![1, 2]).unwrap();

    let result = block_on(super::real_to_complex(
        super::OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");

    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(out.materialize_f64(), vec![(-2.0, 0.0), (3.0, 0.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn minus_typed_sparse_int64_uses_exact_sparse_route() {
    let lhs = SparseTensor::new_integer(
        2,
        1,
        vec![0, 2],
        vec![0, 1],
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
    )
    .unwrap();
    let rhs =
        SparseTensor::new_integer(2, 1, vec![0, 1], vec![0], IntegerStorage::I64(vec![1])).unwrap();
    let Value::SparseTensor(result) = minus_builtin(
        Value::SparseTensor(lhs),
        Value::SparseTensor(rhs),
        Vec::new(),
    )
    .expect("minus") else {
        panic!("expected typed sparse result");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
    );
}

#[test]
fn minus_dense_integer_arrays_preserve_exact_storage() {
    let lhs = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
        .expect("lhs");
    let rhs =
        Tensor::new_integer(IntegerStorage::I64(vec![1, -7, i64::MIN]), vec![1, 3]).expect("rhs");

    let result =
        minus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("integer minus");
    let Value::Tensor(result) = result else {
        panic!("expected integer tensor");
    };
    assert_eq!(result.shape, vec![2, 3]);
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::I64(vec![
            i64::MIN,
            i64::MAX - 1,
            i64::MIN + 7,
            i64::MAX,
            0,
            i64::MAX
        ]))
    );

    let scalar_tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).expect("scalar");
    assert_eq!(
        minus_builtin(Value::Tensor(scalar_tensor), Value::Num(1.0), Vec::new())
            .expect("scalar minus"),
        Value::Int(runmat_value::IntValue::U16(0))
    );
}

#[test]
fn minus_float_arrays_preserve_native_single_class() {
    let lhs = Tensor::from_f32(vec![3.25, -4.0], vec![1, 2]).unwrap();
    let rhs = Tensor::from_f32(vec![2.0, 0.5], vec![1, 2]).unwrap();
    let Value::Tensor(result) =
        minus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
    else {
        panic!("expected single tensor");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![1.25, -4.5])
    );

    let lhs = Tensor::new(vec![0.5, 0.2], vec![1, 2]).unwrap();
    let rhs = Tensor::from_f32(vec![0.2, 0.3], vec![1, 2]).unwrap();
    let Value::Tensor(result) =
        minus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
    else {
        panic!("expected mixed floating tensor");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![
            (0.5_f64 - f64::from(0.2_f32)) as f32,
            (0.2_f64 - f64::from(0.3_f32)) as f32,
        ])
    );
}

#[test]
fn minus_complex_arrays_preserve_native_single_class() {
    let lhs = ComplexTensor::from_f32(vec![(1.25, -2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
    let rhs = Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap();
    let Value::ComplexTensor(result) =
        minus_builtin(Value::ComplexTensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
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
        Some(vec![(0.75_f32, -2.0_f32), (2.0_f32, 4.0_f32)])
    );
}

#[test]
fn minus_mixed_complex_floating_inputs_preserve_order_and_single_class() {
    let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let double = ComplexTensor::new(vec![(3.0, -1.0)], vec![1, 1]).unwrap();
    for (lhs, rhs, expected) in [
        (single.clone(), double.clone(), (-2.0, 3.0)),
        (double.clone(), single.clone(), (2.0, -3.0)),
    ] {
        let result = minus_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex minus");
        let Value::ComplexTensor(result) = result else {
            panic!("expected one-element complex single tensor");
        };
        assert_eq!(
            result.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![expected])
        );
    }
}

#[test]
fn minus_real_complex_single_reverse_path_and_empty_class_are_preserved() {
    let real = Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap();
    let complex = ComplexTensor::from_f32(vec![(1.25, -2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
    let result = minus_builtin(
        Value::Tensor(real),
        Value::ComplexTensor(complex),
        Vec::new(),
    )
    .expect("real-complex minus");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(-0.75, 2.0), (-2.0, -4.0)])
    );
    let lhs = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let rhs = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
    let result = minus_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("empty complex minus");
    let Value::ComplexTensor(result) = result else {
        panic!("expected empty complex single tensor");
    };
    assert_eq!(result.shape, vec![0, 2]);
    assert_eq!(result.as_f32_slice(), Some(&[][..]));
}

#[test]
fn minus_like_complex_conversion_preserves_single_storage() {
    let tensor = Tensor::from_f32(vec![2.0, 3.0], vec![2, 1]).unwrap();
    let result = block_on(super::real_to_complex(
        super::OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");
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
fn minus_rejects_real_integer_with_floating_complex() {
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
        let error = minus_builtin(lhs, rhs, Vec::new()).unwrap_err();
        assert!(error
            .message()
            .contains("complex integer arithmetic is not supported"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_row_column_broadcast() {
    let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    let result = minus_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
        .expect("broadcast minus");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 3]);
            let expected = vec![
                -9.0, -8.0, -7.0, // column-first order
                -19.0, -18.0, -17.0, -29.0, -28.0, -27.0,
            ];
            assert_eq!(t.as_f64_slice().expect("double result"), expected);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_complex_inputs() {
    let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = minus_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex minus");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected = [(-1.0, 3.0), (4.0, -5.0)];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
            }
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_char_input() {
    let chars = CharArray::new("DEF".chars().collect(), 1, 3).unwrap();
    let result =
        minus_builtin(Value::CharArray(chars), Value::Num(1.0), Vec::new()).expect("char minus");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 3]);
            assert_eq!(
                t.as_f64_slice().expect("double result"),
                &[67.0, 68.0, 69.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_logical_input_promotes_to_double() {
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    let tensor = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![2, 2]).unwrap();
    let result = minus_builtin(
        Value::LogicalArray(logical),
        Value::Tensor(tensor),
        Vec::new(),
    )
    .expect("logical");
    match result {
        Value::Tensor(t) => {
            assert_eq!(
                t.as_f64_slice().expect("double result"),
                &[-1.0, -2.0, -2.0, -3.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_dimension_mismatch_errors() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = minus_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).unwrap_err();
    assert!(
        err.message().contains("minus"),
        "unexpected error message: {err}"
    );
}
