use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, SparseTensor, StringArray, Tensor};

#[test]
fn cancelled_invocation_rejects_before_fit_or_output_publication() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = block_on(mnrfit_in_runtime(
        Some(&runtime),
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cancelled invocation cannot publish a model");
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}

fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn mnrfit_admission_preserves_low_rank_predictor_and_response_geometry() {
    let scalar_x = Value::Tensor(Tensor::new(vec![0.0], Vec::new()).unwrap());
    let scalar_y = Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());
    assert_eq!(
        block_on(admission::preflight(
            &scalar_x,
            &scalar_y,
            &mut FitCooperation::new(None),
        ))
        .expect("rank-zero predictors and responses pass geometry preflight"),
        1
    );

    let vector_x = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2]).unwrap());
    let vector_y = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
    let prepared = block_on(admission::prepare(
        vector_x,
        vector_y,
        &mut FitCooperation::new(None),
    ))
    .expect("rank-one predictor and categorical response pass admission");
    assert_eq!(prepared.rows(), 2);

    let nonmatrix_x = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 1, 2]).unwrap());
    let matching_y = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
    let error = block_on(admission::preflight(
        &nonmatrix_x,
        &matching_y,
        &mut FitCooperation::new(None),
    ))
    .expect_err("nonsingleton predictor tail remains invalid");
    assert!(error.message().contains("nonempty numeric matrix"));
}

#[test]
fn mnrfit_rejects_sparse_predictors_and_responses_at_admission() {
    let sparse_x =
        Value::SparseTensor(SparseTensor::new(4, 2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap());
    let sparse_y =
        Value::SparseTensor(SparseTensor::new(4, 3, vec![0, 1, 1, 1], vec![0], vec![1.0]).unwrap());
    let dense_x = tensor_value(vec![0.0; 8], 4, 2);
    let dense_y = tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1);
    let x_error = block_on(admission::prepare(
        sparse_x,
        dense_y,
        &mut FitCooperation::new(None),
    ))
    .err()
    .expect("sparse X is not an admitted predictor matrix");
    assert!(x_error.message().contains("X must be numeric"));
    let y_error = block_on(admission::prepare(
        dense_x,
        sparse_y,
        &mut FitCooperation::new(None),
    ))
    .err()
    .expect("sparse Y is not an admitted response");
    assert!(y_error.message().contains("Y must be"));
}

#[test]
fn mnrfit_binary_logistic_returns_coefficients_stats() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let out = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0, 4.0], 5, 1),
        tensor_value(vec![1.0, 1.0, 2.0, 2.0, 2.0], 5, 1),
        vec![
            Value::String("MaxIter".into()),
            Value::Num(200.0),
            Value::String("TolX".into()),
            Value::Num(1.0e-8),
        ],
    ))
    .unwrap();
    let values = output_list(out);
    let Value::Tensor(b) = &values[0] else {
        panic!("expected B tensor");
    };
    assert_eq!(b.shape(), &[2, 1]);
    assert!(b.materialize_f64()[1] < 0.0);
    assert!(matches!(&values[1], Value::Num(value) if value.is_finite()));
    let Value::Struct(stats) = &values[2] else {
        panic!("expected stats struct");
    };
    assert!(stats.fields.contains_key("se"));
    assert!(stats.fields.contains_key("covb"));
}

#[test]
fn mnrfit_requested_fourth_output_is_runtime_zero_padding() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let values = output_list(
        block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0, 4.0], 5, 1),
            tensor_value(vec![1.0, 1.0, 2.0, 2.0, 2.0], 5, 1),
            Vec::new(),
        ))
        .expect("fourth output padding"),
    );
    assert_eq!(values.len(), 4);
    assert!(matches!(values[3], Value::Num(0.0)));
}

#[test]
fn mnrfit_admits_option_identity_before_resident_download() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
        let y = tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1);
        let err = block_on(mnrfit_builtin(
            Value::GpuTensor(handle.clone()),
            y.clone(),
            vec![Value::from("UnknownOption"), Value::Num(1.0)],
        ))
        .expect_err("unknown option must fail before downloading X");
        assert!(err.message().contains("unknown option 'unknownoption'"));

        let err = block_on(mnrfit_builtin(
            Value::GpuTensor(handle),
            y,
            vec![Value::from("Weights"), Value::Num(1.0), Value::from("TolX")],
        ))
        .expect_err("odd name-value tail must fail before downloading X");
        assert!(err.message().contains("name-value options must be paired"));
    });
}

#[test]
fn mnrfit_inspection_keeps_option_positions_for_compatibility_gates() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1),
        vec![
            Value::from("UnknownOption"),
            Value::Num(1.0),
            Value::from("MaxIter"),
            Value::Num(10.0),
        ],
    ))
    .expect_err("later legacy option retains its original compatibility gate");
    assert_eq!(err.identifier(), LEGACY_OPTIONS_EXTENSION.error_identifier);
}

#[test]
fn mnrfit_positive_integer_rejects_unrepresentable_double_bounds() {
    assert_eq!(positive_integer(&Value::Num(5.0), "MaxIter").unwrap(), 5);
    let boundary = positive_integer(&Value::Num(usize::MAX as f64), "MaxIter");
    if usize::BITS == 64 {
        assert!(boundary.is_err());
    } else {
        assert_eq!(boundary.unwrap(), usize::MAX);
    }
    assert!(positive_integer(&Value::Num((usize::MAX as f64) + 1.0), "MaxIter").is_err());
}

#[test]
fn mnrfit_multiclass_nominal_shape() {
    let out = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0], 6, 1),
        tensor_value(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 6, 1),
        vec![
            Value::String("IterationLimit".into()),
            Value::Num(200.0),
            Value::String("Tolerance".into()),
            Value::Num(1.0e-8),
        ],
    ))
    .unwrap();
    let Value::Tensor(b) = out else {
        panic!("expected B tensor");
    };
    assert_eq!(b.shape(), &[2, 2]);
    assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
}

#[test]
fn mnrfit_binomial_counts_and_weights() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        tensor_value(vec![0.0, 1.0, 3.0, 5.0, 5.0, 4.0, 2.0, 1.0], 4, 2),
        vec![
            Value::String("Weights".into()),
            tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1),
            Value::String("Constant".into()),
            Value::String("on".into()),
        ],
    ))
    .unwrap();
    let Value::Tensor(b) = out else {
        panic!("expected B tensor");
    };
    assert_eq!(b.shape(), &[2, 1]);
    assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
}

#[test]
fn mnrfit_grouped_multinomial_counts() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let out = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        tensor_value(
            vec![8.0, 5.0, 2.0, 1.0, 1.0, 3.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0],
            4,
            3,
        ),
        vec![
            Value::String("IterationLimit".into()),
            Value::Num(200.0),
            Value::String("Tolerance".into()),
            Value::Num(1.0e-8),
        ],
    ))
    .unwrap();
    let values = output_list(out);
    let Value::Tensor(b) = &values[0] else {
        panic!("expected B tensor");
    };
    assert_eq!(b.shape(), &[2, 2]);
    assert!(matches!(&values[1], Value::Num(value) if value.is_finite() && *value >= 0.0));
    let Value::Struct(stats) = &values[2] else {
        panic!("expected stats struct");
    };
    let Value::Tensor(class_names) = stats.fields.get("classNames").unwrap() else {
        panic!("expected numeric class names");
    };
    assert_eq!(class_names.materialize_f64(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn constant_grouped_odds_have_known_intercept_zero_slope_and_zero_deviance() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let values = output_list(
        block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            tensor_value(vec![2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0], 4, 2),
            vec![Value::from("Tolerance"), Value::Num(1.0e-9)],
        ))
        .expect("constant grouped odds have a finite nominal fit"),
    );
    assert_eq!(values.len(), 2);
    let Value::Tensor(beta) = &values[0] else {
        panic!("grouped fit must return a coefficient matrix");
    };
    assert_eq!(beta.shape(), &[2, 1]);
    assert_eq!(beta.numeric_dtype(), runmat_value::NumericDType::F64);
    let coefficients = beta.materialize_f64();
    assert!((coefficients[0] + 3.0f64.ln()).abs() < 1.0e-6);
    assert!(coefficients[1].abs() < 1.0e-6);
    assert!(matches!(&values[1], Value::Num(dev) if dev.abs() < 1.0e-6));
}

#[test]
fn mnrfit_accepts_text_labels_and_omits_missing_rows() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let out = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0, f64::NAN, 2.0, 3.0, 4.0], 6, 1),
        Value::StringArray(
            StringArray::new(
                vec![
                    "low".into(),
                    "high".into(),
                    "low".into(),
                    "low".into(),
                    "high".into(),
                    "high".into(),
                ],
                vec![6, 1],
            )
            .unwrap(),
        ),
        vec![
            Value::String("Options".into()),
            Value::Struct({
                let mut st = runmat_value::UnboundedStructValueBuilder::new();
                st.insert("IterationLimit", Value::Num(200.0));
                st.insert("Tolerance", Value::Num(1.0e-8));
                st.finish()
            }),
        ],
    ))
    .unwrap();
    let values = output_list(out);
    let Value::Struct(stats) = &values[2] else {
        panic!("expected stats struct");
    };
    assert!(
        matches!(stats.fields.get("dfe"), Some(Value::Num(value)) if (*value - 3.0).abs() < 1.0e-8)
    );
    let Value::StringArray(class_names) = stats.fields.get("classNames").unwrap() else {
        panic!("expected text class names");
    };
    assert_eq!(class_names.data(), vec!["high", "low"]);
}

#[test]
fn mnrfit_rejects_unsupported_model() {
    let err = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1),
        vec![
            Value::String("Model".into()),
            Value::String("ordinal".into()),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:mnrfit:InvalidArgument"));
}

#[test]
fn mnrfit_reads_typed_integer_class_vectors_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let out = block_on(mnrfit_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3, 4]), 5, 1),
        poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 2, 2, 2]), 5, 1),
        vec![
            Value::String("Weights".into()),
            poisoned_int_tensor(IntegerStorage::U16(vec![1, 1, 2, 2, 1]), 5, 1),
            Value::String("IterationLimit".into()),
            poisoned_int_tensor(IntegerStorage::U16(vec![200]), 1, 1),
            Value::String("Options".into()),
            Value::Struct({
                let mut st = runmat_value::UnboundedStructValueBuilder::new();
                st.insert(
                    "Tolerance",
                    poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                );
                st.finish()
            }),
        ],
    ))
    .unwrap();
    let values = output_list(out);
    let Value::Tensor(b) = &values[0] else {
        panic!("expected B tensor");
    };
    assert_eq!(b.shape(), &[2, 1]);
    assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
    assert!(matches!(&values[1], Value::Num(value) if value.is_finite()));
}

#[test]
fn mnrfit_reads_typed_integer_grouped_counts_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(mnrfit_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
        poisoned_int_tensor(IntegerStorage::U16(vec![0, 1, 3, 5, 5, 4, 2, 1]), 4, 2),
        vec![
            Value::String("MaxIter".into()),
            poisoned_int_tensor(IntegerStorage::U16(vec![200]), 1, 1),
        ],
    ))
    .unwrap();
    let Value::Tensor(b) = out else {
        panic!("expected B tensor");
    };
    assert_eq!(b.shape(), &[2, 1]);
    assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
}

#[test]
fn mnrfit_integer_roles_and_legacy_options_are_gated() {
    let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 1]), 2, 1);
    let y = tensor_value(vec![1.0, 2.0], 2, 1);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(mnrfit_builtin(x, y.clone(), Vec::new())).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:MnrfitIntegerXExtension")
    );
    let error = block_on(mnrfit_builtin(
        tensor_value(vec![0.0, 1.0], 2, 1),
        y,
        vec![Value::String("MaxIter".into()), Value::Num(10.0)],
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:MnrfitLegacyOptionsExtension")
    );
}

#[test]
fn mnrfit_rejects_lossy_integer_solver_values() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = poisoned_int_tensor(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), 2, 1);
    let error = block_on(mnrfit_builtin(
        x,
        tensor_value(vec![1.0, 2.0], 2, 1),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(error.message.contains("exactly representable"));
}

#[test]
fn mnrfit_integer_boundary_accepts_all_exact_binary64_values() {
    for (integer, storage) in [
        (
            IntValue::I64(1_i64 << 54),
            IntegerStorage::I64(vec![1_i64 << 54]),
        ),
        (IntValue::I64(i64::MIN), IntegerStorage::I64(vec![i64::MIN])),
        (
            IntValue::U64(1_u64 << 54),
            IntegerStorage::U64(vec![1_u64 << 54]),
        ),
    ] {
        assert_eq!(
            exact_integer_as_f64(&integer, "Tolerance").unwrap(),
            integer.to_f64()
        );
        let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        block_on(ensure_exact_integer_tensor(
            &tensor,
            "X",
            &mut FitCooperation::new(None),
        ))
        .unwrap();
    }
    for integer in [
        IntValue::I64((1_i64 << 53) + 1),
        IntValue::U64((1_u64 << 53) + 1),
    ] {
        assert!(exact_integer_as_f64(&integer, "Tolerance").is_err());
    }
}
