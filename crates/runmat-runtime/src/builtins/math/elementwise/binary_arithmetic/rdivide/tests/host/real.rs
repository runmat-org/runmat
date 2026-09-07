use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_scalar_numbers() {
    let result = rdivide_builtin(Value::Num(7.0), Value::Num(2.0), Vec::new()).expect("rdivide");
    match result {
        Value::Num(v) => assert!((v - 3.5).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_integer_arrays_preserve_exact_uint64_values() {
    let lhs = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 3, 0]), vec![1, 3])
        .expect("integer tensor");
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![2, 2, 0]), vec![1, 3])
        .expect("integer tensor");
    let result = rdivide_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new())
        .expect("integer rdivide");
    let Value::Tensor(result) = result else {
        panic!("expected typed integer tensor");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 63, 2, 0]))
    );
}

#[test]
fn rdivide_float_arrays_preserve_native_single_class() {
    let lhs = Tensor::from_f32(vec![9.0, 5.0], vec![1, 2]).unwrap();
    let rhs = Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap();
    let Value::Tensor(result) =
        rdivide_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
    else {
        panic!("expected single tensor");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3.0, 2.5])
    );
}

#[test]
fn rdivide_like_complex_conversion_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I32(vec![-4, 5]), vec![1, 2]).unwrap();

    let result = block_on(real_to_complex(
        OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");

    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(out.materialize_f64(), vec![(-4.0, 0.0), (5.0, 0.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn rdivide_like_complex_conversion_preserves_single_storage() {
    let tensor = Tensor::from_f32(vec![-4.0, 5.0], vec![1, 2]).unwrap();

    let result = block_on(real_to_complex(
        OUTPUT_PROTOTYPE_CONTEXT,
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
        Some(vec![(-4.0, 0.0), (5.0, 0.0)])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_matrix_scalar() {
    let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let result =
        rdivide_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("rdivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(double_values(&t), &[1.0, 2.0, 3.0, 4.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_row_column_broadcast() {
    let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0, 40.0], vec![1, 3]).unwrap();
    let result = rdivide_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
        .expect("broadcast rdivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 3]);
            let expected = [0.1, 0.2, 0.3, 0.05, 0.10, 0.15, 0.025, 0.05, 0.075];
            for (got, exp) in double_values(&t).iter().zip(expected.iter()) {
                assert!((got - exp).abs() < EPS);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_division_by_zero() {
    let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
    let result =
        rdivide_builtin(Value::Tensor(tensor), Value::Num(0.0), Vec::new()).expect("rdivide");
    match result {
        Value::Tensor(t) => {
            let values = double_values(&t);
            assert!(values[0].is_nan());
            assert!(values[1].is_infinite());
            assert!(values[2].is_infinite());
            assert!(values[2].is_sign_negative());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_logical_inputs_promote() {
    let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
    let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![2, 2]).unwrap();
    let result = rdivide_builtin(
        Value::LogicalArray(logical),
        Value::Tensor(tensor),
        Vec::new(),
    )
    .expect("logical rdivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [1.0, 0.0, 0.25, 0.125];
            for (got, exp) in double_values(&t).iter().zip(expected.iter()) {
                assert!((got - exp).abs() < EPS);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_char_array_promotes_to_double() {
    let chars = CharArray::new_row("AB");
    let result =
        rdivide_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("rdivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert!((double_values(&t)[0] - 32.5).abs() < EPS);
            assert!((double_values(&t)[1] - 33.0).abs() < EPS);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_same_class_integer_inputs_preserve_class() {
    let lhs = Value::Int(IntValue::I32(6));
    let rhs = Value::Int(IntValue::I32(4));
    let result = rdivide_builtin(lhs, rhs, Vec::new()).expect("rdivide");
    assert_eq!(result, Value::Int(IntValue::I32(2)));
}
