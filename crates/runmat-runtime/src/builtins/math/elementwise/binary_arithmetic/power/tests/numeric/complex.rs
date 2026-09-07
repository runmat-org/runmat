use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_complex_scalar() {
    let result = power_builtin(
        Value::Complex(1.0, 2.0),
        Value::Complex(0.5, -1.0),
        Vec::new(),
    )
    .expect("power");
    match result {
        Value::Complex(re, im) => {
            assert!((re - 4.382565059863358).abs() < 1e-9);
            assert!((im + 1.1243974773611554).abs() < 1e-9);
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}

#[test]
fn power_preserves_native_and_mixed_complex_single_storage() {
    let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let single_exponent = ComplexTensor::from_f32(vec![(2.0, 0.0)], vec![1, 1]).unwrap();
    let result = power_builtin(
        Value::ComplexTensor(single.clone()),
        Value::ComplexTensor(single_exponent),
        Vec::new(),
    )
    .expect("complex single power");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    let value = result.as_f32_slice().expect("complex single")[0];
    assert!((value.0 + 3.0).abs() < 1e-5);
    assert!((value.1 - 4.0).abs() < 1e-5);

    let double_exponent = ComplexTensor::new(vec![(2.0, 0.0)], vec![1, 1]).unwrap();
    let result = power_builtin(
        Value::ComplexTensor(single),
        Value::ComplexTensor(double_exponent),
        Vec::new(),
    )
    .expect("mixed complex power");
    let Value::ComplexTensor(result) = result else {
        panic!("expected mixed complex result in single");
    };
    let value = result.as_f32_slice().expect("complex single")[0];
    assert!((value.0 + 3.0).abs() < 1e-5);
    assert!((value.1 - 4.0).abs() < 1e-5);
}

#[test]
fn power_mixed_real_complex_single_paths_preserve_single() {
    let complex_base = ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, 0.0)], vec![1, 2]).unwrap();
    let real_exponent = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let result = power_builtin(
        Value::ComplexTensor(complex_base),
        Value::Tensor(real_exponent),
        Vec::new(),
    )
    .expect("complex-real power");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    let values = result.as_f32_slice().expect("complex single");
    assert!((values[0].0 - 0.0).abs() < 1e-5);
    assert!((values[0].1 - 2.0).abs() < 1e-5);
    assert!((values[1].0 - 8.0).abs() < 1e-5);

    let real_base = Tensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap();
    let complex_exponent =
        ComplexTensor::from_f32(vec![(2.0, 0.0), (3.0, 0.0)], vec![1, 2]).unwrap();
    let result = power_builtin(
        Value::Tensor(real_base),
        Value::ComplexTensor(complex_exponent),
        Vec::new(),
    )
    .expect("real-complex power");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(16.0, 0.0), (8.0, 0.0)])
    );
}

#[test]
fn power_preserves_empty_complex_single_class() {
    let base = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let exponent = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
    let result = power_builtin(
        Value::ComplexTensor(base),
        Value::ComplexTensor(exponent),
        Vec::new(),
    )
    .expect("complex power");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(result.shape, vec![0, 2]);
    assert_eq!(result.as_f32_slice(), Some(&[][..]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_zero_complex_positive_real_part() {
    let result =
        power_builtin(Value::Num(0.0), Value::Complex(1.0, 2.0), Vec::new()).expect("power");
    match result {
        Value::Complex(re, im) => {
            assert!(re.abs() < 1e-12);
            assert!(im.abs() < 1e-12);
        }
        other => panic!("expected zero complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_zero_complex_negative_real_part() {
    let result =
        power_builtin(Value::Num(0.0), Value::Complex(-1.0, 1.0), Vec::new()).expect("power");
    match result {
        Value::Complex(re, im) => {
            assert!(re.is_infinite());
            assert!(im.is_nan());
        }
        other => panic!("expected complex infinity, got {other:?}"),
    }
}
