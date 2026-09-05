use super::*;

#[test]
fn scalar_hypot_value_leaves_typed_integer_tensor_for_storage_dispatch() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("integer tensor");

    assert_eq!(scalar_hypot_value(&Value::Tensor(tensor)), None);
}

#[test]
fn scalar_hypot_value_leaves_complex_tensor_for_storage_dispatch() {
    let storage =
        IntegerComplexStorage::new(IntegerStorage::I16(vec![3]), IntegerStorage::I16(vec![4]))
            .expect("complex integer storage");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");

    assert_eq!(scalar_hypot_value(&Value::ComplexTensor(tensor)), None);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_typed_integer_tensor_broadcast_reads_integer_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let matrix = Tensor::new_integer(IntegerStorage::I16(vec![3, 4, 5, 12]), vec![2, 2]).unwrap();
    let row = Tensor::new_integer(IntegerStorage::I16(vec![4, 3]), vec![1, 2]).unwrap();

    let result = hypot_builtin(Value::Tensor(matrix), Value::Tensor(row)).expect("integer hypot");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [
                3.0_f64.hypot(4.0),
                4.0_f64.hypot(4.0),
                5.0_f64.hypot(3.0),
                12.0_f64.hypot(3.0),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn hypot_preserves_native_single_real_complex_mixed_and_empty_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let left = Tensor::from_f32(vec![3.0, 5.0], vec![2, 1]).unwrap();
    let right = Tensor::from_f32(vec![4.0, 12.0], vec![2, 1]).unwrap();
    let Value::Tensor(output) =
        hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("single hypot")
    else {
        panic!("expected single tensor");
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![5.0, 13.0])
    );

    let left = Tensor::new(vec![3.0, 5.0], vec![2, 1]).unwrap();
    let right = Tensor::from_f32(vec![4.0, 12.0], vec![2, 1]).unwrap();
    let Value::Tensor(output) =
        hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("mixed hypot")
    else {
        panic!("expected mixed result to use double");
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F64(vec![5.0, 13.0])
    );

    let left = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_992, 3]),
        vec![1, 2],
    )
    .unwrap();
    let right = Tensor::from_f32(vec![1.0, 4.0], vec![1, 2]).unwrap();
    let Value::Tensor(output) =
        hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("integer/single hypot")
    else {
        panic!("expected integer/single result to use double");
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
    assert_eq!(
        output.materialize_f64(),
        vec![(9_007_199_254_740_992_f64).hypot(1.0), 5.0]
    );

    let complex = ComplexTensor::from_f32(vec![(3.0, 4.0)], vec![1, 1]).unwrap();
    let real = Tensor::from_f32(vec![12.0], vec![1, 1]).unwrap();
    let Value::Tensor(output) = hypot_builtin(Value::ComplexTensor(complex), Value::Tensor(real))
        .expect("complex single hypot")
    else {
        panic!("one-element single result must retain tensor class");
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![13.0])
    );

    let left = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
    let right = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let Value::Tensor(output) =
        hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("empty hypot")
    else {
        panic!("expected empty mixed tensor");
    };
    assert_eq!(output.shape, vec![0, 3]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F64(Vec::new())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_complex_scalars() {
    let left = (3.0, 4.0);
    let right = (-1.0, 2.0);
    let result = hypot_builtin(
        Value::Complex(left.0, left.1),
        Value::Complex(right.0, right.1),
    )
    .expect("complex hypot");
    let expected = complex_magnitude(left.0, left.1).hypot(complex_magnitude(right.0, right.1));
    match result {
        Value::Num(v) => assert!((v - expected).abs() < 1e-12),
        other => panic!("expected scalar norm, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_complex_tensor_with_real() {
    let complex = ComplexTensor::new(vec![(3.0, 4.0), (5.0, 12.0)], vec![2, 1]).unwrap();
    let real = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
    let result = hypot_builtin(Value::ComplexTensor(complex), Value::Tensor(real)).expect("mixed");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 1]);
            let expected = [
                complex_magnitude(3.0, 4.0).hypot(0.0),
                complex_magnitude(5.0, 12.0).hypot(1.0),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_char_array_inputs() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
    let result =
        hypot_builtin(Value::CharArray(chars), Value::Int(IntValue::I32(1))).expect("char hypot");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected = [
                (65.0f64.powi(2) + 1.0).sqrt(),
                (66.0f64.powi(2) + 1.0).sqrt(),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12);
            }
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_logical_inputs() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical array");
    let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let result =
        hypot_builtin(Value::LogicalArray(logical), Value::Tensor(tensor)).expect("logical");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![2, 2]);
            let expected = [
                1.0_f64.hypot(0.0),
                0.0_f64.hypot(1.0),
                0.0_f64.hypot(2.0),
                1.0_f64.hypot(3.0),
            ];
            for (actual, expect) in out.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
            }
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}
