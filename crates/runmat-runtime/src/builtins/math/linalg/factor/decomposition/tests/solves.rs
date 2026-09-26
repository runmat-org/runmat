use super::*;

#[test]
fn all_integer_classes_and_logical_coefficients_solve_only_in_runmat_mode() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_coefficient_storages() {
        let dtype = storage.numeric_dtype();
        let class = storage.class_name();
        let d = call_constructor(vec![typed_int_tensor(storage, vec![2, 2])])
            .unwrap_or_else(|error| panic!("{class} construction failed: {error}"));
        let stored = object_to_spec(&d).expect("stored decomposition").matrix;
        assert!(
            matches!(stored, Value::Tensor(ref tensor) if tensor.integer_storage().is_some_and(|storage| storage.numeric_dtype() == dtype)),
            "{class} must remain exact before solve"
        );
        let result = call_mldivide(d, tensor(&[8.0, 12.0], 2, 1))
            .unwrap_or_else(|error| panic!("{class} solve failed: {error}"));
        assert_eq!(approx_vec(result), vec![4.0, 3.0], "{class}");
    }

    let logical = Value::LogicalArray(
        LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical identity"),
    );
    let d = call_constructor(vec![logical]).expect("logical decomposition");
    assert!(matches!(
        object_to_spec(&d).expect("stored decomposition").matrix,
        Value::LogicalArray(_)
    ));
    assert_eq!(
        approx_vec(call_mldivide(d, tensor(&[8.0, 12.0], 2, 1)).expect("logical solve")),
        vec![8.0, 12.0]
    );
}

#[test]
fn integer_decomposition_right_solve_promotes_at_the_solve_boundary() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let d = call_constructor(vec![typed_int_tensor(
        IntegerStorage::U64(vec![2, 0, 0, 4]),
        vec![2, 2],
    )])
    .expect("integer decomposition");
    let result = call_mrdivide(tensor(&[8.0, 12.0], 1, 2), d).expect("right solve");
    let Value::Tensor(result) = result else {
        panic!("expected promoted double tensor")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F64);
    assert_eq!(result.materialize_f64(), vec![4.0, 3.0]);
}

#[test]
fn decomposition_left_solve_reuses_mldivide_semantics() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mldivide(d, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 4.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);
}

#[test]
fn decomposition_right_solve_uses_rhs_object() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 1, 2);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mrdivide(b, d).expect("right solve");

    let values = approx_vec(result);
    assert!((values[0] - 4.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);
}

#[test]
fn scalar_multiply_and_divide_adjust_solve_matrix() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");
    let scaled = call_mtimes(Value::Num(2.0), d).expect("scale");
    let divided = call_rdivide(scaled, Value::Num(4.0)).expect("divide");

    let result = call_mldivide(divided, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 8.0).abs() < 1e-12);
    assert!((values[1] - 6.0).abs() < 1e-12);
}

#[test]
fn ctranspose_toggles_property_and_solves_transposed_system() {
    let a = tensor(&[1.0, 0.0, 2.0, 1.0], 2, 2);
    let b = tensor(&[3.0, 1.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");
    let dt = call_ctranspose(d).expect("ctranspose");

    let prop = block_on(decomposition_subsref(
        dt.clone(),
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(IS_CONJUGATE_TRANSPOSED_FIELD.to_string()),
    ))
    .expect("property");
    assert_eq!(prop, Value::Bool(true));

    let result = call_mldivide(dt, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 3.0).abs() < 1e-12);
    assert!((values[1] + 5.0).abs() < 1e-12);
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
