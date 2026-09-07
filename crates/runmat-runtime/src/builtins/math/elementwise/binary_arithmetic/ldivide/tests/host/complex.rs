use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_complex_inputs() {
    let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = ldivide_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex ldivide");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected = [(0.0, -1.0), (-0.28, -0.04)];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < 1e-10 && (got.1 - exp.1).abs() < 1e-10);
            }
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn ldivide_preserves_native_complex_single_storage() {
    let divisor = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let numerator = ComplexTensor::from_f32(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = ldivide_builtin(
        Value::ComplexTensor(divisor),
        Value::ComplexTensor(numerator),
        Vec::new(),
    )
    .expect("complex ldivide");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.0, -1.0), (-0.28, -0.04)])
    );
}

#[test]
fn ldivide_mixed_complex_floating_inputs_return_single_without_scalar_collapse() {
    let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let double = ComplexTensor::new(vec![(2.0, -1.0)], vec![1, 1]).unwrap();
    for (divisor, numerator, expected) in [
        (single.clone(), double.clone(), (0.0, -1.0)),
        (double.clone(), single.clone(), (0.0, 1.0)),
    ] {
        let result = ldivide_builtin(
            Value::ComplexTensor(divisor),
            Value::ComplexTensor(numerator),
            Vec::new(),
        )
        .expect("complex ldivide");
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
fn ldivide_mixed_real_complex_single_paths_preserve_single() {
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (-3.0, 1.0)], vec![1, 2]).unwrap();
    let real = Tensor::new(vec![2.0, 0.5], vec![1, 2]).unwrap();

    let result = ldivide_builtin(
        Value::ComplexTensor(complex.clone()),
        Value::Tensor(real.clone()),
        Vec::new(),
    )
    .expect("complex divisor ldivide");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.4, -0.8), (-0.15, -0.05)])
    );
    let result = ldivide_builtin(
        Value::Tensor(real),
        Value::ComplexTensor(complex),
        Vec::new(),
    )
    .expect("complex numerator ldivide");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.5, 1.0), (-6.0, 2.0)])
    );
}

#[test]
fn ldivide_preserves_empty_complex_single_class() {
    let divisor = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let numerator = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
    let result = ldivide_builtin(
        Value::ComplexTensor(divisor),
        Value::ComplexTensor(numerator),
        Vec::new(),
    )
    .expect("complex ldivide");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(result.shape, vec![0, 2]);
    assert_eq!(result.as_f32_slice(), Some(&[][..]));
}
