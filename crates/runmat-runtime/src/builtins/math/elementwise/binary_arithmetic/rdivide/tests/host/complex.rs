use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_complex_inputs() {
    let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = rdivide_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex rdivide");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected = [(0.0, 1.0), (-3.5, 0.5)];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < 1e-10 && (got.1 - exp.1).abs() < 1e-10);
            }
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn rdivide_preserves_native_complex_single_storage() {
    let lhs = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
    let rhs = ComplexTensor::from_f32(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
    let result = rdivide_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex rdivide");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.0, 1.0), (-3.5, 0.5)])
    );
}

#[test]
fn rdivide_mixed_complex_floating_inputs_return_single_without_scalar_collapse() {
    let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let double = ComplexTensor::new(vec![(2.0, -1.0)], vec![1, 1]).unwrap();
    for (lhs, rhs, expected) in [
        (single.clone(), double.clone(), (0.0, 1.0)),
        (double.clone(), single.clone(), (0.0, -1.0)),
    ] {
        let result = rdivide_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex rdivide");
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
fn rdivide_mixed_real_complex_single_paths_preserve_single() {
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (-3.0, 1.0)], vec![1, 2]).unwrap();
    let real = Tensor::new(vec![2.0, 0.5], vec![1, 2]).unwrap();

    let result = rdivide_builtin(
        Value::ComplexTensor(complex.clone()),
        Value::Tensor(real.clone()),
        Vec::new(),
    )
    .expect("complex-real rdivide");
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
    let result = rdivide_builtin(
        Value::Tensor(real),
        Value::ComplexTensor(complex),
        Vec::new(),
    )
    .expect("real-complex rdivide");
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
}

#[test]
fn rdivide_preserves_empty_complex_single_class() {
    let lhs = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let rhs = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
    let result = rdivide_builtin(
        Value::ComplexTensor(lhs),
        Value::ComplexTensor(rhs),
        Vec::new(),
    )
    .expect("complex rdivide");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(result.shape, vec![0, 2]);
    assert_eq!(result.as_f32_slice(), Some(&[][..]));
}
