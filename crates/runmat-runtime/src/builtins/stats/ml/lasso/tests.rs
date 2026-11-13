use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage};
use runmat_value::{StringArray, StructValue};

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn output_pair(value: Value) -> (Value, Value) {
    let Value::OutputList(values) = value else {
        panic!("expected output list");
    };
    (values[0].clone(), values[1].clone())
}

fn row_field<'a>(info: &'a StructValue, name: &str) -> &'a Tensor {
    let Some(Value::Tensor(tensor)) = info.fields.get(name) else {
        panic!("expected tensor field {name}");
    };
    tensor
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn lasso_admission_and_vector_parsers_preserve_low_rank_geometry() {
    let scalar_x = tensor(vec![2.0], Vec::new());
    assert_eq!(predictor_matrix_dimensions(&scalar_x).unwrap(), (1, 1));
    let vector_x = tensor(vec![1.0, 2.0], vec![2]);
    assert_eq!(predictor_matrix_dimensions(&vector_x).unwrap(), (2, 1));
    let nonmatrix_x = tensor(vec![1.0, 2.0], vec![1, 1, 2]);
    assert!(predictor_matrix_dimensions(&nonmatrix_x).is_err());

    let scalar_y = tensor(vec![3.0], Vec::new());
    assert_eq!(value_to_real_vector("y", scalar_y).unwrap(), vec![3.0]);
    let vector_y = tensor(vec![3.0, 4.0], vec![2]);
    assert_eq!(value_to_real_vector("y", vector_y).unwrap(), vec![3.0, 4.0]);

    let lambda_scalar = tensor(vec![0.25], Vec::new());
    assert_eq!(
        numeric_vector(&lambda_scalar, "Lambda").unwrap(),
        vec![0.25]
    );
    let lambda_vector = tensor(vec![0.25, 0.5], vec![2]);
    assert_eq!(
        numeric_vector(&lambda_vector, "Lambda").unwrap(),
        vec![0.25, 0.5]
    );
    let lambda_matrix = tensor(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
    assert!(numeric_vector(&lambda_matrix, "Lambda").is_err());
}

#[test]
fn explicit_lambda_recovers_sparse_coefficients_with_intercept() {
    let x = tensor(
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, //
            1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![5, 2],
    );
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0, 9.0], vec![5, 1]);
    let out = block_on(lasso_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0], vec![1, 1]),
            Value::CharArray(CharArray::new_row("Standardize")),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let Value::Tensor(coeffs) = out else {
        panic!("expected coefficient tensor");
    };
    assert_eq!(coeffs.shape(), &[2, 1]);
    assert!((coeffs.materialize_f64()[0] - 2.0).abs() < 1.0e-8);
    assert!(coeffs.materialize_f64()[1].abs() < 1.0e-8);
}

#[test]
fn malformed_use_covariance_does_not_silently_become_false() {
    let result = block_on(lasso_builtin(
        tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
        tensor(vec![1.0, 3.0, 5.0], vec![3, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("UseCovariance".into()),
            Value::String("not-a-boolean".into()),
        ]),
    ));
    let error = result.expect_err("unknown boolean spelling must reject");
    assert_eq!(error.identifier(), Some("RunMat:lasso:InvalidArgument"));
}

#[test]
fn predictor_geometry_rejects_nonsingleton_tail_and_keeps_singleton_tail() {
    let rejected = block_on(lasso_builtin(
        tensor(vec![1.0; 8], vec![2, 2, 2]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    assert_eq!(
        rejected
            .expect_err("third predictor axis must not be ignored")
            .identifier(),
        Some("RunMat:lasso:InvalidArgument")
    );
    let accepted = block_on(lasso_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 0.0], vec![4, 2, 1]),
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Lambda"),
            Value::Num(0.1),
        ]),
    ));
    assert!(
        accepted.is_ok(),
        "singleton tail retains matrix geometry: {accepted:?}"
    );
}

#[test]
fn invalid_host_option_precedes_unregistered_predictor_download() {
    let resident = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("unregistered resident fixture")
        .handle;
    let error = block_on(lasso_builtin(
        Value::GpuTensor(resident),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Alpha"),
            Value::Num(2.0),
        ]),
    ))
    .expect_err("host option must reject before provider access");
    assert_eq!(error.identifier(), Some("RunMat:lasso:InvalidArgument"));
}

#[test]
fn alpha_weights_and_fitinfo_output_are_supported() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let x = tensor(
        vec![
            0.0, 1.0, 2.0, 3.0, //
            0.0, 0.0, 1.0, 1.0,
        ],
        vec![4, 2],
    );
    let y = tensor(vec![0.0, 1.0, 3.0, 4.0], vec![4, 1]);
    let out = block_on(lasso_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0, 0.2], vec![1, 2]),
            Value::CharArray(CharArray::new_row("Alpha")),
            Value::Num(0.5),
            Value::CharArray(CharArray::new_row("Weights")),
            tensor(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1]),
            Value::CharArray(CharArray::new_row("PredictorNames")),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap();
    let (b, fit_info) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected coefficients");
    };
    assert_eq!(coeffs.shape(), &[2, 2]);
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo struct");
    };
    assert!(info.fields.contains_key("Intercept"));
    assert!(info.fields.contains_key("Lambda"));
    assert!(info.fields.contains_key("DF"));
    assert!(info.fields.contains_key("MSE"));
    assert!(info.fields.contains_key("PredictorNames"));
}

#[test]
fn lasso_reads_typed_integer_storage_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]);
    let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), vec![4, 1]);
    let out = block_on(lasso_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            Value::CharArray(CharArray::new_row("Alpha")),
            cleared_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            Value::CharArray(CharArray::new_row("Weights")),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1]), vec![4, 1]),
            Value::CharArray(CharArray::new_row("MaxIter")),
            cleared_int_tensor(IntegerStorage::U16(vec![50]), vec![1, 1]),
            Value::CharArray(CharArray::new_row("Standardize")),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let Value::Tensor(coeffs) = out else {
        panic!("expected coefficient tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 1]);
    assert!((coeffs.materialize_f64()[0] - 2.0).abs() < 1.0e-8);
}

#[test]
fn lasso_gates_integer_roles_before_gather_and_rejects_lossy_data() {
    let integer_data = cleared_int_tensor(IntegerStorage::U8(vec![1, 2]), vec![2, 1]);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = ensure_extensions(&integer_data, &tensor(vec![1.0, 2.0], vec![2, 1]), &[])
        .expect_err("strict mode must reject integer data before gather");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:LassoIntegerDataExtension")
    );
    let err = ensure_extensions(
        &tensor(vec![1.0, 2.0], vec![2, 1]),
        &tensor(vec![1.0, 2.0], vec![2, 1]),
        &admit_options(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("MaxIter"), Value::Int(IntValue::U16(20))],
        ))
        .unwrap(),
    )
    .expect_err("strict mode must reject integer controls before gather");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:LassoIntegerControlExtension")
    );
    drop(_strict);

    let wide = cleared_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
    let err = value_to_real_tensor("X", wide).expect_err("lossy integer must reject");
    assert!(err.message.contains("exactly representable"));
}

#[test]
fn lasso_positive_usize_parses_integer_bounds_exactly() {
    assert_eq!(
        positive_usize(&Value::Int(IntValue::U16(3)), "NumLambda").unwrap(),
        3
    );
    assert_eq!(
        positive_usize(
            &cleared_int_tensor(IntegerStorage::U64(vec![3]), vec![1, 1]),
            "NumLambda"
        )
        .unwrap(),
        3
    );
    for value in [
        Value::Int(IntValue::I8(-1)),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(positive_usize(&value, "NumLambda").is_err());
    }
    assert!(bounded_usize(
        &cleared_int_tensor(IntegerStorage::U64(vec![usize::MAX as u64]), vec![1, 1]),
        "NumLambda",
        MAX_NUM_LAMBDA
    )
    .is_err());
    assert_eq!(
        bounded_usize(
            &Value::Int(IntValue::U16(MAX_NUM_LAMBDA as u16)),
            "NumLambda",
            MAX_NUM_LAMBDA
        )
        .unwrap(),
        MAX_NUM_LAMBDA
    );
}

#[test]
fn cross_validation_adds_minimum_error_fields() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let x = tensor(
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, //
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ],
        vec![6, 2],
    );
    let y = tensor(vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0], vec![6, 1]);
    let out = block_on(lasso_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0, 0.1, 0.3], vec![1, 3]),
            Value::CharArray(CharArray::new_row("CV")),
            Value::Num(3.0),
        ]),
    ))
    .unwrap();
    let (_, fit_info) = output_pair(out);
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo");
    };
    assert!(info.fields.contains_key("SE"));
    assert!(info.fields.contains_key("LambdaMinMSE"));
    assert!(info.fields.contains_key("Lambda1SE"));
    assert!(info.fields.contains_key("IndexMinMSE"));
    assert!(info.fields.contains_key("Index1SE"));
}

#[test]
fn dfmax_filters_cv_fields_consistently() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let x = tensor(
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, //
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ],
        vec![6, 2],
    );
    let y = tensor(vec![0.0, 2.0, 4.0, 7.0, 8.0, 11.0], vec![6, 1]);
    let out = block_on(lasso_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0, 10.0], vec![1, 2]),
            Value::CharArray(CharArray::new_row("DFmax")),
            Value::Num(1.0),
            Value::CharArray(CharArray::new_row("CV")),
            Value::Num(3.0),
        ]),
    ))
    .unwrap();
    let (b, fit_info) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected coefficients");
    };
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo");
    };
    let lambda = row_field(&info, "Lambda");
    let mse = row_field(&info, "MSE");
    let se = row_field(&info, "SE");
    assert_eq!(coeffs.cols(), lambda.materialize_f64().len());
    assert_eq!(mse.materialize_f64().len(), lambda.materialize_f64().len());
    assert_eq!(se.materialize_f64().len(), lambda.materialize_f64().len());
    let Some(Value::Num(index_min_mse)) = info.fields.get("IndexMinMSE") else {
        panic!("expected IndexMinMSE");
    };
    let Some(Value::Num(index_1se)) = info.fields.get("Index1SE") else {
        panic!("expected Index1SE");
    };
    assert!(*index_min_mse >= 1.0 && *index_min_mse <= lambda.materialize_f64().len() as f64);
    assert!(*index_1se >= 1.0 && *index_1se <= lambda.materialize_f64().len() as f64);
}

#[test]
fn omitted_predictor_names_and_use_covariance_report_actual_state() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let x = tensor(
        vec![
            0.0, 1.0, 2.0, 3.0, //
            1.0, 1.0, 1.0, 1.0,
        ],
        vec![4, 2],
    );
    let y = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let out = block_on(lasso_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.1], vec![1, 1]),
            Value::CharArray(CharArray::new_row("UseCovariance")),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    let (_, fit_info) = output_pair(out);
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo");
    };
    let Some(Value::Cell(names)) = info.fields.get("PredictorNames") else {
        panic!("expected PredictorNames cell");
    };
    assert_eq!(names.data().len(), 0);
    assert_eq!(names.shape, vec![1, 0]);
    let Some(Value::LogicalArray(use_covariance)) = info.fields.get("UseCovariance") else {
        panic!("expected UseCovariance logical");
    };
    assert_eq!(use_covariance.data, vec![0]);
}

#[test]
fn matrix_shaped_lambda_and_weights_are_rejected() {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = tensor(vec![1.0, 2.0], vec![2, 1]);
    let err = block_on(lasso_builtin(
        x.clone(),
        y.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0, 0.1, 0.2, 0.3], vec![2, 2]),
        ]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lasso:InvalidArgument"));

    let err = block_on(lasso_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::CharArray(CharArray::new_row("Weights")),
            tensor(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]),
        ]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lasso:InvalidArgument"));
}

#[test]
fn validation_mse_uses_heldout_weights() {
    let x = Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap();
    let weighted = validation_mse(&x, &[1.0, 3.0], &[0.0], 0.0, &[0, 1], Some(&[9.0, 1.0]));
    assert!((weighted - 1.8).abs() < 1.0e-12);
}

#[test]
fn invalid_dimensions_error() {
    let err = block_on(lasso_builtin(
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lasso:InvalidArgument"));
}
