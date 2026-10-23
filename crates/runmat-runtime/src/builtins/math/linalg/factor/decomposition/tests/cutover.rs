use super::*;

#[test]
fn cholesky_validation_is_independent_of_requested_outputs() {
    for count in [0, 1] {
        let _outputs =
            crate::output_count::push_output_count(Some(count)).expect("output-count scope");
        let error = call_constructor(vec![
            tensor(&[1.0, 2.0, 2.0, 1.0], 2, 2),
            Value::from("chol"),
        ])
        .expect_err("non-positive-definite coefficient must fail even for a discarded object");
        assert_eq!(
            error.identifier(),
            Some("RunMat:decomposition:InvalidInput")
        );
        assert!(error.message().contains("positive definite"));
        call_constructor(vec![
            tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2),
            Value::from("chol"),
        ])
        .expect("valid positive definite matrix");
    }
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = call_constructor(vec![
        tensor(&[1.0, 2.0, 2.0, 1.0], 2, 2),
        Value::from("chol"),
    ])
    .expect_err("extra outputs fail before factorization");
    assert!(error.message().contains("at most one output"));
}

#[test]
fn every_method_checks_output_count_before_inputs_or_provider_access() {
    let d = call_constructor(vec![tensor(&[2.0], 1, 1)]).unwrap();
    let providerless = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let errors = [
        call_constructor(vec![providerless.clone()]).unwrap_err(),
        block_on(super::super::decomposition_subsref(
            d.clone(),
            Value::Num(0.0),
        ))
        .unwrap_err(),
        block_on(decomposition_uplus(d.clone())).unwrap_err(),
        block_on(decomposition_uminus(d.clone())).unwrap_err(),
        call_ctranspose(d.clone()).unwrap_err(),
        block_on(decomposition_times(d.clone(), providerless.clone())).unwrap_err(),
        call_mtimes(d.clone(), providerless.clone()).unwrap_err(),
        call_rdivide(d.clone(), providerless.clone()).unwrap_err(),
        call_mrdivide(providerless.clone(), d.clone()).unwrap_err(),
        call_mldivide(d, providerless).unwrap_err(),
    ];
    for error in errors {
        assert_eq!(
            error.identifier(),
            Some("RunMat:decomposition:InvalidInput")
        );
        assert!(error.message().contains("at most one output"));
    }
}

#[test]
fn empty_and_nd_constructor_boundaries_are_preserved() {
    let d = call_constructor(vec![tensor(&[], 0, 3)]).expect("empty rectangular matrix");
    let spec = object_to_spec(&d).unwrap();
    assert_eq!(matrix_dims(&spec.matrix), (0, 3));
    let trailing_singleton =
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]).unwrap());
    let d = call_constructor(vec![trailing_singleton])
        .expect("normalized matrix-compatible singleton tail");
    assert_eq!(matrix_dims(&object_to_spec(&d).unwrap().matrix), (2, 2));
    let nd = Value::Tensor(Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap());
    assert!(call_constructor(vec![nd])
        .unwrap_err()
        .message()
        .contains("two-dimensional"));
}

#[test]
fn scalar_right_division_and_numeric_right_solve_have_distinct_outputs() {
    let d = call_constructor(vec![tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2)]).unwrap();
    assert!(matches!(
        call_mrdivide(d.clone(), Value::Num(2.0)).unwrap(),
        Value::Object(_)
    ));
    assert_eq!(
        approx_vec(call_rdivide(tensor(&[8.0, 12.0], 1, 2), d).unwrap()),
        vec![4.0, 3.0]
    );
}
