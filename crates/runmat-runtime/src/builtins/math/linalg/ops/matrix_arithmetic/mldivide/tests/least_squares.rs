use super::*;

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
    assert_eq!(gathered.shape(), [2, 1]);

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
fn tall_sparse_qr_reuses_csc_and_preserves_native_rhs_precision() {
    let _accel_guard = test_support::accel_test_lock();
    clear_accel_provider_state();
    let context = RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let solve = |lhs, rhs| {
        block_on(super::super::mldivide_builtin(
            Some(context.clone()),
            lhs,
            rhs,
        ))
    };
    // A = [1 0; 0 1; 1 1], held as native CSC.
    let sparse = runmat_value::SparseTensor::new_f32(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 2],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    // The second column is inconsistent, exercising a nonzero least-squares
    // residual rather than only an exact solution of an overdetermined system.
    let rhs = Tensor::from_f32(vec![2.0, 3.0, 5.0, -1.0, 4.0, 4.0], vec![3, 2]).unwrap();
    let Value::Tensor(solution) = solve(Value::SparseTensor(sparse), Value::Tensor(rhs)).unwrap()
    else {
        panic!("sparse QR must return a native single tensor");
    };
    assert_eq!(solution.shape(), [2, 2]);
    for (actual, expected) in
        solution
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip([2.0_f32, 3.0, -2.0 / 3.0, 13.0 / 3.0])
    {
        assert!((actual - expected).abs() < 1e-5);
    }

    let complex = runmat_value::SparseTensor::new_complex(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 2],
        vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
    )
    .unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, 1.0), (3.0, -2.0), (5.0, -1.0)], vec![3, 1]).unwrap();
    let Value::ComplexTensor(solution) =
        solve(Value::SparseTensor(complex), Value::ComplexTensor(rhs)).unwrap()
    else {
        panic!("sparse QR must return a native complex tensor");
    };
    for (actual, expected) in solution
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip([(2.0, 1.0), (3.0, -2.0)])
    {
        assert!((actual.0 - expected.0).abs() < 1e-12);
        assert!((actual.1 - expected.1).abs() < 1e-12);
    }
}
