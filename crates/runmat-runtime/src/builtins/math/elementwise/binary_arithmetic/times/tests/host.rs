use super::*;

#[test]
fn times_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = TIMES_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = times(A, B)"));
    assert!(labels.contains(&"C = times(A, B, \"like\", prototype)"));
}

#[test]
fn times_parser_error_has_stable_identifier() {
    let err = times_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
        .expect_err("expected parser error");
    assert_eq!(err.identifier(), TIMES_ERROR_INVALID_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_scalar_numbers() {
    let result = times_builtin(Value::Num(2.0), Value::Num(3.5), Vec::new()).expect("times");
    match result {
        Value::Num(v) => assert!((v - 7.0).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_matrix_scalar() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = times_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("times");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn times_preserves_single_for_single_and_mixed_floating_inputs() {
    let single_lhs = Tensor::from_f32(vec![1.5, -2.0], vec![2, 1]).unwrap();
    let single_rhs = Tensor::from_f32(vec![2.0, 4.0], vec![2, 1]).unwrap();
    let result = times_builtin(
        Value::Tensor(single_lhs.clone()),
        Value::Tensor(single_rhs),
        Vec::new(),
    )
    .expect("single times single");
    let Value::Tensor(result) = result else {
        panic!("expected tensor result");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3.0, -8.0])
    );

    for (lhs, rhs) in [
        (
            Value::Tensor(single_lhs.clone()),
            Value::Tensor(Tensor::new(vec![0.2, 0.5], vec![2, 1]).unwrap()),
        ),
        (
            Value::Tensor(Tensor::new(vec![0.2, 0.5], vec![2, 1]).unwrap()),
            Value::Tensor(single_lhs.clone()),
        ),
    ] {
        let result = times_builtin(lhs, rhs, Vec::new()).expect("mixed floating times");
        let Value::Tensor(result) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.3, -1.0])
        );
    }
}

#[test]
fn times_like_complex_conversion_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).unwrap();

    let result = block_on(super::real_to_complex(
        super::OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");

    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(out.materialize_f64(), vec![(2.0, 0.0), (3.0, 0.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn times_typed_sparse_uint64_uses_exact_sparse_route() {
    let lhs = SparseTensor::new_integer(
        2,
        1,
        vec![0, 2],
        vec![0, 1],
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    )
    .unwrap();
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![2, 1]).unwrap();
    let Value::SparseTensor(result) =
        times_builtin(Value::SparseTensor(lhs), Value::Tensor(rhs), Vec::new()).expect("times")
    else {
        panic!("expected typed sparse result");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![18_014_398_509_481_986, u64::MAX]))
    );
}

#[test]
fn times_dense_integer_arrays_preserve_exact_storage_without_mirror() {
    let lhs = Tensor::new_integer(
        IntegerStorage::U64(vec![(1_u64 << 63) + 1, u64::MAX]),
        vec![2, 1],
    )
    .expect("lhs");
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 0]), vec![1, 3]).expect("rhs");

    let result =
        times_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("integer times");
    let Value::Tensor(result) = result else {
        panic!("expected integer tensor");
    };
    assert_eq!(result.shape, vec![2, 3]);
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            (1_u64 << 63) + 1,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            0,
            0
        ]))
    );

    let scalar_tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![i16::MAX]), vec![1, 1]).expect("scalar");
    assert_eq!(
        times_builtin(Value::Tensor(scalar_tensor), Value::Num(2.0), Vec::new())
            .expect("scalar times"),
        Value::Int(IntValue::I16(i16::MAX))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_row_column_broadcast() {
    let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    let result = times_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
        .expect("broadcast times");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 3]);
            let expected = vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0];
            assert_eq!(t.materialize_f64(), expected);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_complex_inputs() {
    let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = times_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex times");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected = [(4.0, 3.0), (1.0, 7.0)];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
            }
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn times_preserves_native_complex_single_storage() {
    let lhs = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let rhs = ComplexTensor::from_f32(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = times_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex times");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(4.0, 3.0), (1.0, 7.0)])
    );
}

#[test]
fn times_mixed_complex_floating_inputs_return_single() {
    let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let double = ComplexTensor::new(vec![(2.0, -1.0)], vec![1, 1]).unwrap();
    for (lhs, rhs) in [
        (single.clone(), double.clone()),
        (double.clone(), single.clone()),
    ] {
        let result = times_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex times");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            result.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![(4.0, 3.0)])
        );
    }
}

#[test]
fn times_mixed_real_complex_single_paths_preserve_single() {
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (-3.0, 1.0)], vec![1, 2]).unwrap();
    let real = Tensor::new(vec![2.0, 0.5], vec![1, 2]).unwrap();
    for (lhs, rhs) in [
        (
            Value::ComplexTensor(complex.clone()),
            Value::Tensor(real.clone()),
        ),
        (
            Value::Tensor(real.clone()),
            Value::ComplexTensor(complex.clone()),
        ),
    ] {
        let result = times_builtin(lhs, rhs, Vec::new()).expect("complex-real times");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            result.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![(2.0, 4.0), (-1.5, 0.5)])
        );
    }
}

#[test]
fn times_preserves_empty_complex_single_class() {
    let lhs = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let rhs = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
    let result = times_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex times");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(result.shape, vec![0, 2]);
    assert_eq!(result.as_f32_slice(), Some(&[][..]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_char_input() {
    let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
    let result =
        times_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("char times");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 3]);
            assert_eq!(t.materialize_f64(), vec![130.0, 132.0, 134.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_logical_input_promotes_to_double() {
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    let tensor = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![2, 2]).unwrap();
    let result = times_builtin(
        Value::LogicalArray(logical),
        Value::Tensor(tensor),
        Vec::new(),
    )
    .expect("logical");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![2.0, 0.0, 3.0, 0.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_dimension_mismatch_errors() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = times_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).unwrap_err();
    assert!(err.message().contains("times"), "unexpected error: {err}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_same_class_integer_inputs_preserve_class() {
    let lhs = Value::Int(IntValue::I32(3));
    let rhs = Value::Int(IntValue::I32(5));
    let result = times_builtin(lhs, rhs, Vec::new()).expect("times");
    assert_eq!(result, Value::Int(IntValue::I32(15)));
}
