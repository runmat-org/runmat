use super::*;

#[test]
fn chol_checks_exact_integer_and_complex_integer_boundary_before_materialization() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for input in [
        typed_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        typed_int_tensor(IntegerStorage::I64(vec![i64::MAX]), vec![1, 1]),
        mirrorless_complex_int_tensor(
            IntegerStorage::I64(vec![4]),
            IntegerStorage::I64(vec![0]),
            vec![1, 1],
        ),
    ] {
        let error = call_constructor(vec![input, Value::from("chol")])
            .expect_err("opaque storage permission does not authorize rounded Cholesky arithmetic");
        assert_eq!(
            error.identifier(),
            Some("RunMat:decomposition:InvalidInput")
        );
        assert!(error.message().contains("positive definite"));
    }
    call_constructor(vec![
        typed_int_tensor(IntegerStorage::U64(vec![1_u64 << 53]), vec![1, 1]),
        Value::from("chol"),
    ])
    .expect("the exact binary64 boundary remains admitted");
    call_constructor(vec![
        typed_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        Value::from("diagonal"),
    ])
    .expect("structural construction still stores wide integers exactly");
}

#[test]
fn triangular_flag_ignores_opposite_triangle() {
    let a = tensor(&[2.0, 99.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![
        a,
        Value::String("triangular".to_string().into()),
        Value::String("upper".to_string().into()),
    ])
    .expect("decomposition");

    let result = call_mldivide(d, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 4.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);
}

#[test]
fn chol_type_rejects_non_positive_definite_matrix() {
    let a = tensor(&[1.0, 0.0, 0.0, -1.0], 2, 2);

    let err = call_constructor(vec![a, Value::String("chol".to_string().into())])
        .expect_err("chol error");

    assert!(err.message().contains("positive definite"));
}

#[test]
fn chol_lower_flag_uses_lower_triangle_as_hermitian_matrix() {
    let a = tensor(&[4.0, 1.0, 99.0, 3.0], 2, 2);
    let b = tensor(&[5.0, 4.0], 2, 1);
    let d = call_constructor(vec![
        a,
        Value::String("chol".to_string().into()),
        Value::String("lower".to_string().into()),
    ])
    .expect("chol decomposition");

    let result = call_mldivide(d, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 1.0).abs() < 1e-12);
    assert!((values[1] - 1.0).abs() < 1e-12);
}

#[test]
fn sparse_inputs_construct_reusable_factors_for_the_public_solve() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0])
        .expect("sparse");

    let factor = call_constructor(vec![Value::SparseTensor(sparse)]).expect("sparse factor");
    let answer = call_mldivide(factor, tensor(&[3.0, 8.0], 2, 1)).expect("sparse solve");
    assert_eq!(approx_vec(answer), vec![3.0, 4.0]);
}

#[test]
fn unsupported_decomposition_types_are_rejected() {
    let a = tensor(&[1.0, 0.0, 0.0, 1.0], 2, 2);

    let err =
        call_constructor(vec![a, Value::String("banded".to_string().into())]).expect_err("type");

    assert!(err.message().contains("unsupported decomposition type"));
}
