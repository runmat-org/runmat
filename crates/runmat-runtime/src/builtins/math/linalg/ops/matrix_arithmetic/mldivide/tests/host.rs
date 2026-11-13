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
    assert_eq!(gathered.shape(), [2, 1]);

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
fn solves_native_precision_sparse_triangles_without_densifying_coefficients() {
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
    let lower = runmat_value::SparseTensor::new_f32(
        2,
        2,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![2.0, 1.0, 3.0],
    )
    .unwrap();
    let rhs = Tensor::from_f32(vec![2.0, 4.0, 4.0, 8.0], vec![2, 2]).unwrap();
    let Value::Tensor(solution) = solve(Value::SparseTensor(lower), Value::Tensor(rhs)).unwrap()
    else {
        panic!("sparse single triangular solve must retain a dense single result");
    };
    assert_eq!(solution.as_f32_slice(), Some(&[1.0, 1.0, 2.0, 2.0][..]));

    let empty_lower = runmat_value::SparseTensor::new_f32(
        2,
        2,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![2.0, 1.0, 3.0],
    )
    .unwrap();
    let empty_rhs = Tensor::from_f32(Vec::new(), vec![2, 0]).unwrap();
    let Value::Tensor(empty) =
        solve(Value::SparseTensor(empty_lower), Value::Tensor(empty_rhs)).unwrap()
    else {
        panic!("zero-column RHS must remain a native single tensor");
    };
    assert_eq!(empty.shape(), [2, 0]);

    let upper =
        runmat_value::SparseTensor::new(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![2.0, 1.0, 3.0])
            .unwrap();
    let rhs = Tensor::new(vec![3.0, 3.0], vec![2, 1]).unwrap();
    let Value::Tensor(solution) = solve(Value::SparseTensor(upper), Value::Tensor(rhs)).unwrap()
    else {
        panic!("sparse upper triangular solve must retain a dense result");
    };
    assert_eq!(solution.as_f64_slice(), Some(&[1.0, 1.0][..]));

    let complex = runmat_value::SparseTensor::new_complex(
        2,
        2,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![(2.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    )
    .unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, 0.0), (1.0, 2.0)], vec![2, 1]).unwrap();
    let Value::ComplexTensor(solution) =
        solve(Value::SparseTensor(complex), Value::ComplexTensor(rhs)).unwrap()
    else {
        panic!("sparse complex triangular solve must retain a complex result");
    };
    let values = solution.as_f64_slice().unwrap();
    assert!((values[0].0 - 1.0).abs() < 1e-12);
    assert!(values[0].1.abs() < 1e-12);
    assert!((values[1].0 - 1.0).abs() < 1e-12);
    assert!(values[1].1.abs() < 1e-12);

    let scalar_matrix =
        runmat_value::SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap();
    assert!(matches!(
        solve(Value::SparseTensor(scalar_matrix), Value::Num(4.0)).unwrap(),
        Value::Num(2.0)
    ));
    let scalar_complex =
        runmat_value::SparseTensor::new_complex(1, 1, vec![0, 1], vec![0], vec![(2.0, 0.0)])
            .unwrap();
    assert!(matches!(
        solve(
            Value::SparseTensor(scalar_complex),
            Value::Complex(4.0, 2.0)
        )
        .unwrap(),
        Value::Complex(2.0, 1.0)
    ));

    let complex_single =
        runmat_value::SparseTensor::new_complex_f32(1, 1, vec![0, 1], vec![0], vec![(2.0, 0.0)])
            .unwrap();
    let single_rhs = ComplexTensor::from_f32(vec![(4.0, 2.0)], vec![1, 1]).unwrap();
    let Value::ComplexTensor(single) = solve(
        Value::SparseTensor(complex_single),
        Value::ComplexTensor(single_rhs),
    )
    .unwrap() else {
        panic!("complex single solve must retain its native class");
    };
    assert_eq!(
        single.as_f32_slice().unwrap()[0],
        runmat_value::ComplexElement(2.0, 1.0)
    );
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

            let a_mat = DMatrix::from_column_slice(a.rows(), a.cols(), &mat_a);
            let b_mat = DMatrix::from_column_slice(b.rows(), b.cols(), &mat_b);
            let x_mat = DMatrix::from_column_slice(out.rows(), out.cols(), &mat_x);
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
