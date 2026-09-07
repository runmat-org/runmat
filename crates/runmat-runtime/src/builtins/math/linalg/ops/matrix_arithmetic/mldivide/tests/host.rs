use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn solves_square_system() {
    let _accel_guard = test_support::accel_test_lock();
    clear_accel_provider_state();
    let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
    let result =
        mldivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("mldivide");
    let gathered = test_support::gather(result).expect("gather");
    assert_eq!(gathered.shape, vec![2, 1]);

    let mat_a = DMatrix::from_column_slice(a.rows(), a.cols(), &a.materialize_f64());
    let mat_x = DMatrix::from_column_slice(
        gathered.rows(),
        gathered.cols(),
        &gathered.materialize_f64(),
    );
    let mat_b = DMatrix::from_column_slice(b.rows(), b.cols(), &b.materialize_f64());
    let residual = &mat_a * &mat_x - mat_b;
    assert!(residual.norm() < 1e-12);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn solves_least_squares() {
    let _accel_guard = test_support::accel_test_lock();
    clear_accel_provider_state();
    let a = Tensor::new(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap();
    let b = Tensor::new(vec![7.0, 8.0, 9.0], vec![3, 1]).unwrap();
    let result =
        mldivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("mldivide");
    let gathered = test_support::gather(result).expect("gather");
    assert_eq!(gathered.shape, vec![2, 1]);

    let mat_a = DMatrix::from_column_slice(a.rows(), a.cols(), &a.materialize_f64());
    let mat_x = DMatrix::from_column_slice(
        gathered.rows(),
        gathered.cols(),
        &gathered.materialize_f64(),
    );
    let mat_b = DMatrix::from_column_slice(b.rows(), b.cols(), &b.materialize_f64());
    let residual = &mat_a * &mat_x - mat_b;
    assert!(residual.norm() < 1e-10);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn supports_complex_inputs() {
    let a = ComplexTensor::new(
        vec![(2.0, 1.0), (-1.0, 0.0), (1.0, -2.0), (3.0, -2.0)],
        vec![2, 2],
    )
    .unwrap();
    let b = ComplexTensor::new(vec![(1.0, 0.0), (4.0, 1.0)], vec![2, 1]).unwrap();
    let result = mldivide_builtin(
        Value::ComplexTensor(a.clone()),
        Value::ComplexTensor(b.clone()),
    )
    .expect("mldivide");
    match result {
        Value::ComplexTensor(out) => {
            let mat_a: Vec<Complex64> = a
                .materialize_f64()
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect();
            let mat_b: Vec<Complex64> = b
                .materialize_f64()
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect();
            let mat_x: Vec<Complex64> = out
                .materialize_f64()
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect();

            let a_mat = DMatrix::from_column_slice(a.rows, a.cols, &mat_a);
            let b_mat = DMatrix::from_column_slice(b.rows, b.cols, &mat_b);
            let x_mat = DMatrix::from_column_slice(out.rows, out.cols, &mat_x);
            let residual = &a_mat * &x_mat - b_mat;
            assert!(residual.norm() < 1e-6, "residual = {}", residual);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn reports_dimension_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = unwrap_error(mldivide_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err());
    assert_eq!(err.identifier(), MLDIVIDE_ERROR_INVALID_INPUT.identifier);
    assert!(
        err.message().contains("Matrix dimensions must agree"),
        "unexpected error message: {err}"
    );
}
