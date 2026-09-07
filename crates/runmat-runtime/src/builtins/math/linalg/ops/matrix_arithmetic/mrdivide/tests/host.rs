use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn solves_square_system() {
    let _accel_guard = test_support::accel_test_lock();
    clear_accel_provider_state();
    let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let result = mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).expect("mrdivide");
    let gathered = test_support::gather(result).expect("gather");
    let expected = vec![3.0, 2.0, -2.0, -1.0];
    assert_eq!(gathered.shape, vec![2, 2]);
    for (val, exp) in gathered.materialize_f64().iter().zip(expected.into_iter()) {
        assert!((val - exp).abs() < 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn solves_least_squares() {
    let _accel_guard = test_support::accel_test_lock();
    clear_accel_provider_state();
    let a = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
    let result =
        mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("mrdivide");
    let gathered = test_support::gather(result).expect("gather");
    let expected = host_mrdivide_real(&a, &b);
    assert_eq!(gathered.shape, expected.shape);
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert!(
            (actual - expected).abs() < 1e-10,
            "actual={actual} expected={expected}"
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn supports_complex_inputs() {
    let a = ComplexTensor::new(
        vec![(1.0, 2.0), (5.0, 6.0), (3.0, -4.0), (7.0, -2.0)],
        vec![2, 2],
    )
    .unwrap();
    let b = ComplexTensor::new(
        vec![(2.0, -1.0), (1.0, 0.5), (0.5, 1.0), (3.0, 2.0)],
        vec![2, 2],
    )
    .unwrap();
    let result =
        mrdivide_builtin(Value::ComplexTensor(a), Value::ComplexTensor(b)).expect("mrdivide");
    match result {
        Value::ComplexTensor(out) => {
            let expected = [
                (-0.7902439, 1.28780488),
                (-0.72780488, 3.2897561),
                (0.48780488, -1.6097561),
                (2.0097561, -2.31219512),
            ];
            for (value, (er, ei)) in out.materialize_f64().iter().zip(expected.into_iter()) {
                let (vr, vi) = *value;
                assert!((vr - er).abs() < 1e-6);
                assert!((vi - ei).abs() < 1e-6);
            }
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn reports_dimension_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = unwrap_error(mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err());
    assert_eq!(err.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
    assert!(
        err.message().contains("Matrix dimensions must agree"),
        "unexpected error message: {err}"
    );
}
