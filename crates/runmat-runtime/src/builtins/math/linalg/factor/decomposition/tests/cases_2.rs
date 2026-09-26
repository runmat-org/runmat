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
fn complex_decomposition_solves_complex_rhs() {
    let a = complex_tensor(&[(2.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 2.0)], 2, 2);
    let b = complex_tensor(&[(8.0, 2.0), (4.0, 2.0)], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mldivide(d, b).expect("complex solve");

    match result {
        Value::ComplexTensor(tensor) => {
            assert!((tensor.materialize_f64()[0].0 - 4.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[0].1 - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[1].0 - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[1].1 + 2.0).abs() < 1e-12);
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}

#[test]
fn registered_constructor_and_ctranspose_dispatch_work_together() {
    let a = tensor(&[1.0, 0.0, 2.0, 1.0], 2, 2);
    let d = block_on(crate::call_builtin_async("decomposition", &[a])).expect("constructor");
    let dt = block_on(crate::call_builtin_async("ctranspose", &[d])).expect("dispatch");
    let b = tensor(&[3.0, 1.0], 2, 1);

    let result = call_mldivide(dt, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 3.0).abs() < 1e-12);
    assert!((values[1] + 5.0).abs() < 1e-12);
}

#[test]
fn decomposition_scalar_operands_read_typed_integer_storage_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let d = call_constructor(vec![a]).expect("decomposition");
    let scale = Tensor::new_integer(IntegerStorage::I16(vec![3]), vec![1, 1]).expect("scale");

    let scaled = call_mtimes(d, Value::Tensor(scale.clone())).expect("scale");
    let prop = block_on(decomposition_subsref(
        scaled,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(SCALE_FACTOR_FIELD.to_string()),
    ))
    .expect("ScaleFactor");
    assert_eq!(approx_vec(prop), vec![3.0]);
    assert_eq!(
        complex_scalar(&Value::Tensor(scale), "scale").unwrap(),
        (3.0, 0.0)
    );

    let storage = runmat_value::IntegerComplexStorage::new(
        IntegerStorage::I16(vec![4]),
        IntegerStorage::I16(vec![-2]),
    )
    .expect("typed complex storage");
    let complex_scale = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex scale");
    assert_eq!(
        complex_scalar(&Value::ComplexTensor(complex_scale), "scale").unwrap(),
        (4.0, -2.0)
    );
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
fn private_matrix_storage_is_not_publicly_readable() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let d = call_constructor(vec![tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2)]).expect("decomposition");

    let subsref_err = block_on(decomposition_subsref(
        d.clone(),
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(MATRIX_FIELD.to_string()),
    ))
    .expect_err("private subsref should fail");
    assert!(subsref_err
        .message()
        .contains("unknown decomposition property"));

    let getfield_err = block_on(crate::call_builtin_async(
        "getfield",
        &[d, Value::String(MATRIX_FIELD.to_string())],
    ))
    .expect_err("private getfield should fail");
    assert!(matches!(
        getfield_err.identifier(),
        Some("RunMat:PropertyPrivateAccess")
            | Some("RunMat:compatibility:GetfieldObjectFamilyExtension")
    ));
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

#[test]
fn ctranspose_updates_rectangular_matrix_size_property() {
    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let d = call_constructor(vec![a]).expect("decomposition");
    let dt = call_ctranspose(d).expect("ctranspose");

    let size = block_on(decomposition_subsref(
        dt,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(MATRIX_SIZE_FIELD.to_string()),
    ))
    .expect("MatrixSize");

    assert_eq!(approx_vec(size), vec![3.0, 2.0]);
}

#[test]
fn complex_scale_marks_decomposition_nonreal() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let d = call_constructor(vec![a]).expect("decomposition");
    let scaled = call_mtimes(Value::Complex(0.0, 1.0), d).expect("complex scale");

    let is_real = block_on(decomposition_subsref(
        scaled,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(IS_REAL_FIELD.to_string()),
    ))
    .expect("IsReal");

    assert_eq!(is_real, Value::Bool(false));
}

#[test]
fn complex_single_decomposition_reports_single_datatype() {
    let matrix = Value::ComplexTensor(
        ComplexTensor::from_f32(
            vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (4.0, -1.0)],
            vec![2, 2],
        )
        .expect("complex single matrix"),
    );
    let d = call_constructor(vec![matrix]).expect("complex single decomposition");
    let datatype = block_on(decomposition_subsref(
        d,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(DATATYPE_FIELD.to_string()),
    ))
    .expect("Datatype");
    assert_eq!(datatype, Value::String("single".to_string()));
}

#[test]
fn decomposition_scaling_preserves_complex_single_storage_through_solve() {
    let coefficient = Value::ComplexTensor(
        ComplexTensor::from_f32(
            vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (4.0, -1.0)],
            vec![2, 2],
        )
        .expect("single coefficient"),
    );
    let rhs = Value::ComplexTensor(
        ComplexTensor::from_f32(vec![(8.0, 2.0), (12.0, -3.0)], vec![2, 1]).expect("single rhs"),
    );
    let d = call_constructor(vec![coefficient]).expect("decomposition");
    let scaled = call_mtimes(Value::Complex(2.0, -1.0), d).expect("scale");
    let Value::ComplexTensor(result) = call_mldivide(scaled, rhs).expect("solve") else {
        panic!("expected complex tensor")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);

    let real_single = Value::Tensor(
        Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("single matrix"),
    );
    let Value::ComplexTensor(scaled) =
        scale_matrix(real_single, (1.0, 1.0)).expect("complex scale")
    else {
        panic!("expected complex tensor")
    };
    assert_eq!(scaled.numeric_dtype(), NumericDType::F32);
}
