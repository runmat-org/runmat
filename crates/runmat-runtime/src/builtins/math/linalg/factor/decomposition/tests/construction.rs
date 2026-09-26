use super::*;

#[test]
fn triangular_flag_ignores_opposite_triangle() {
    let a = tensor(&[2.0, 99.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![
        a,
        Value::String("triangular".to_string()),
        Value::String("upper".to_string()),
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

    let err = call_constructor(vec![a, Value::String("chol".to_string())]).expect_err("chol error");

    assert!(err.message().contains("positive definite"));
}

#[test]
fn chol_lower_flag_uses_lower_triangle_as_hermitian_matrix() {
    let a = tensor(&[4.0, 1.0, 99.0, 3.0], 2, 2);
    let b = tensor(&[5.0, 4.0], 2, 1);
    let d = call_constructor(vec![
        a,
        Value::String("chol".to_string()),
        Value::String("lower".to_string()),
    ])
    .expect("chol decomposition");

    let result = call_mldivide(d, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 1.0).abs() < 1e-12);
    assert!((values[1] - 1.0).abs() < 1e-12);
}

#[test]
fn sparse_inputs_are_rejected_until_sparse_solves_exist() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0])
        .expect("sparse");

    let err = call_constructor(vec![Value::SparseTensor(sparse)]).expect_err("sparse reject");

    assert!(err.message().contains("sparse coefficient matrices"));
}

#[test]
fn unsupported_decomposition_types_are_rejected() {
    let a = tensor(&[1.0, 0.0, 0.0, 1.0], 2, 2);

    let err = call_constructor(vec![a, Value::String("banded".to_string())]).expect_err("type");

    assert!(err.message().contains("unsupported decomposition type"));
}
