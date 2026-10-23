use super::super::values::ensure_exact_fitlm_integer_scalar_boundary;
use super::*;

fn build_fit_spec(
    first: Value,
    rest: Vec<Value>,
    arguments: FitlmArguments,
) -> BuiltinResult<FitSpec> {
    super::super::build_fit_spec(
        first,
        crate::call::FunctionArgumentView::from_slice(&rest),
        arguments,
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fitlm_dense_matrix_projection_preserves_rank_zero_and_rank_one_policy() {
    let scalar = Tensor::new(vec![7.0], Vec::new()).expect("rank-zero tensor");
    assert_eq!(fitlm_matrix_dimensions(&scalar), (0, 0));

    let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("rank-one tensor");
    assert_eq!(fitlm_matrix_dimensions(&vector), (1, 3));

    let one_response = Value::Tensor(Tensor::new(vec![4.0], vec![1, 1]).expect("one response"));
    preflight_response_shape(
        &Value::Tensor(vector.clone()),
        crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&one_response)),
    )
    .expect("rank-one predictors have one observation at preflight");
    let rank_zero_error = preflight_response_shape(
        &Value::Tensor(scalar.clone()),
        crate::call::FunctionArgumentView::from_slice(&[one_response]),
    )
    .expect_err("rank-zero predictors have no observation at preflight");
    assert!(rank_zero_error.message().contains("length(y)"));

    let arguments = || FitlmArguments {
        option_start: 1,
        options: Vec::new(),
    };
    let spec = build_fit_spec(
        Value::Tensor(vector.clone()),
        vec![Value::Tensor(
            Tensor::new(vec![4.0], vec![1, 1]).expect("one response"),
        )],
        arguments(),
    )
    .expect("rank-one predictors are one observation with three predictors");
    assert_eq!(fitlm_matrix_dimensions(&spec.x), (1, 3));
    assert_eq!(spec.y, vec![4.0]);
    assert_eq!(
        spec.predictor_names,
        vec!["x1".to_string(), "x2".to_string(), "x3".to_string()]
    );

    let mismatch = build_fit_spec(
        Value::Tensor(vector),
        vec![Value::Tensor(
            Tensor::new(vec![4.0, 5.0], vec![2, 1]).expect("two responses"),
        )],
        arguments(),
    )
    .expect_err("rank-one predictors do not have two observations");
    assert!(mismatch.message().contains("length(y)"));

    let phantom_row = build_fit_spec(
        Value::Tensor(scalar),
        vec![Value::Tensor(
            Tensor::new(vec![4.0], vec![1, 1]).expect("one response"),
        )],
        arguments(),
    )
    .expect_err("rank-zero predictors do not acquire a phantom observation");
    assert!(phantom_row.message().contains("length(y)"));
}

#[test]
fn fitlm_header_keeps_model_spec_separate_from_typed_options() {
    let first = tensor_value(vec![0.0, 1.0], 2, 1);
    let values = vec![
        tensor_value(vec![1.0, 3.0], 2, 1),
        Value::from("linear"),
        Value::from("Intercept"),
        Value::Bool(false),
    ];
    let header = HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&values),
    )
    .unwrap()
    .validate(values.len())
    .unwrap();
    assert_eq!(header.option_start, 2);
    assert_eq!(header.options.len(), 1);
    assert_eq!(header.options[0].name.identity, FitlmOption::Intercept);
    assert_eq!(header.options[0].name.index, 2);

    let invalid = vec![
        tensor_value(vec![1.0, 3.0], 2, 1),
        Value::from("linear"),
        Value::from("RobustOpts"),
        Value::Bool(true),
    ];
    assert!(HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&invalid)
    )
    .unwrap()
    .validate(invalid.len())
    .is_err());
}

#[test]
fn fitlm_rejects_known_rank_before_materialization() {
    let first = Value::Tensor(Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap());
    assert!(preflight_matrix_rank(&first).is_err());
}

#[test]
fn fitlm_preflights_host_option_values_and_response_shape() {
    let first = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let values = vec![
        tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1),
        Value::from("Intercept"),
        Value::Bool(false),
    ];
    let prepared = HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&values),
    )
    .unwrap()
    .validate(values.len())
    .unwrap()
    .preflight_values(crate::call::FunctionArgumentView::from_slice(&values))
    .unwrap();
    assert!(matches!(
        prepared.options[0].predecoded,
        Some(FitlmAction::Intercept(false))
    ));
    let invalid = vec![
        values[0].clone(),
        Value::from("Intercept"),
        Value::from("bad"),
    ];
    assert!(HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&invalid)
    )
    .unwrap()
    .validate(invalid.len())
    .unwrap()
    .preflight_values(crate::call::FunctionArgumentView::from_slice(&invalid))
    .err()
    .expect("invalid Intercept")
    .message()
    .contains("Intercept"));
    let short = vec![tensor_value(vec![1.0, 3.0, 5.0], 3, 1)];
    assert!(preflight_response_shape(
        &first,
        crate::call::FunctionArgumentView::from_slice(&short)
    )
    .unwrap_err()
    .message()
    .contains("length(y)"));
}

#[test]
fn logical_predictors_follow_the_numeric_matrix_training_path() {
    let x =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![0, 0, 1, 1], vec![4, 1]).unwrap());
    let y = tensor_value(vec![1.0, 1.0, 3.0, 3.0], 4, 1);
    let model = block_on(fitlm_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![y]),
    ))
    .unwrap();
    let predicted = block_on(predict_builtin(
        model,
        tensor_value(vec![0.0, 1.0], 2, 1),
        Vec::new(),
    ))
    .unwrap();
    let values = tensor(&predicted).materialize_f64();
    assert_eq!(values.len(), 2);
    assert!((values[0] - 1.0).abs() < 1.0e-10);
    assert!((values[1] - 3.0).abs() < 1.0e-10);
}

#[test]
fn fitlm_matrix_fits_coefficients_and_predicts() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![y]),
    ))
    .unwrap();
    let model_object = object(model.clone());
    assert_eq!(
        model_object.properties().get("NumObservations"),
        Some(&Value::Num(4.0))
    );
    let coeffs = match model_object.properties().get("Coefficients").unwrap() {
        Value::Object(object) => table_variables(object).unwrap(),
        other => panic!("coefficients {other:?}"),
    };
    let estimates = tensor(coeffs.fields.get("Estimate").unwrap());
    assert!((estimates.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!((estimates.materialize_f64()[1] - 2.0).abs() < 1.0e-10);
    let predicted = block_on(predict_builtin(
        model,
        tensor_value(vec![4.0, 5.0], 2, 1),
        Vec::new(),
    ))
    .unwrap();
    let ypred = tensor(&predicted);
    assert_eq!(ypred.shape(), [2, 1]);
    assert!((ypred.materialize_f64()[0] - 9.0).abs() < 1.0e-10);
    assert!((ypred.materialize_f64()[1] - 11.0).abs() < 1.0e-10);
}

#[test]
fn registered_predict_preserves_statistical_output_overloads_without_ambient_arity() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![y]),
    ))
    .unwrap();
    let args = [model, tensor_value(vec![4.0, 5.0], 2, 1)];
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("predict");
    assert_eq!(bindings.len(), 1);
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _ambient_outputs =
        crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let predicted = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime.clone()), Some(1)),
    ))
    .unwrap();
    assert_eq!(
        predicted.kind(),
        runmat_value::ValueSequenceKind::CommaSeparated
    );
    assert_eq!(predicted.len(), 1);
    let values = tensor(&predicted.as_slice()[0]);
    assert_eq!(values.shape(), [2, 1]);
    assert!((values.materialize_f64()[0] - 9.0).abs() < 1.0e-10);

    let predicted = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime), Some(2)),
    ))
    .unwrap();
    assert_eq!(
        predicted.kind(),
        runmat_value::ValueSequenceKind::CommaSeparated
    );
    assert_eq!(predicted.len(), 2);
}

#[test]
fn fitlm_and_predict_read_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let model = block_on(fitlm_builtin(
        None,
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 99]), 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 999]), 4, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1]), 4, 1),
            Value::from("Exclude"),
            poisoned_int_tensor(IntegerStorage::U8(vec![4]), 1, 1),
            Value::from("Intercept"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
        ]),
    ))
    .unwrap();
    let predicted = block_on(predict_builtin(
        model,
        poisoned_int_tensor(IntegerStorage::I16(vec![3, 4]), 2, 1),
        Vec::new(),
    ))
    .unwrap();
    let ypred = tensor(&predicted);
    assert_eq!(ypred.shape(), [2, 1]);
    assert!((ypred.materialize_f64()[0] - 7.0).abs() < 1.0e-10);
    assert!((ypred.materialize_f64()[1] - 9.0).abs() < 1.0e-10);
}

#[test]
fn fitlm_runmat_mode_admits_all_integer_classes_by_role() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_fitlm_integer_storages() {
        let model = block_on(fitlm_builtin(
            None,
            poisoned_int_tensor(storage.clone(), 3, 1),
            crate::builtins::common::test_support::builtin_arguments(vec![
                poisoned_int_tensor(storage.clone(), 3, 1),
                Value::from("Weights"),
                poisoned_int_tensor(storage, 3, 1),
                Value::from("Exclude"),
                poisoned_int_tensor(IntegerStorage::U8(Vec::new()), 0, 1),
                Value::from("Intercept"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            ]),
        ))
        .expect("all-class integer fitlm roles");
        assert!(matches!(model, Value::Object(_)));
    }
}

#[test]
fn fitlm_strict_mode_rejects_each_integer_role_independently() {
    let x = || tensor_value(vec![1.0, 2.0, 3.0], 3, 1);
    let y = || tensor_value(vec![1.0, 2.0, 3.0], 3, 1);
    let integer = || poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), 3, 1);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let cases = [
        (
            block_on(fitlm_builtin(
                None,
                integer(),
                crate::builtins::common::test_support::builtin_arguments(vec![y()]),
            ))
            .unwrap_err(),
            "RunMat:compatibility:FitlmIntegerPredictorExtension",
        ),
        (
            block_on(fitlm_builtin(
                None,
                x(),
                crate::builtins::common::test_support::builtin_arguments(vec![integer()]),
            ))
            .unwrap_err(),
            "RunMat:compatibility:FitlmIntegerResponseExtension",
        ),
        (
            block_on(fitlm_builtin(
                None,
                x(),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    y(),
                    Value::from("Weights"),
                    integer(),
                ]),
            ))
            .unwrap_err(),
            "RunMat:compatibility:FitlmIntegerWeightsExtension",
        ),
        (
            block_on(fitlm_builtin(
                None,
                x(),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    y(),
                    Value::from("Exclude"),
                    poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                ]),
            ))
            .unwrap_err(),
            "RunMat:compatibility:FitlmIntegerSelectorExtension",
        ),
        (
            block_on(fitlm_builtin(
                None,
                x(),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    y(),
                    Value::from("Intercept"),
                    poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                ]),
            ))
            .unwrap_err(),
            "RunMat:compatibility:FitlmIntegerControlExtension",
        ),
    ];
    for (error, identifier) in cases {
        assert_eq!(error.identifier(), Some(identifier));
    }
}

#[test]
fn fitlm_rejects_inexact_floating_roles_but_keeps_selectors_structural() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    ensure_exact_fitlm_integer_scalar_boundary(NumericScalar::U64(1_u64 << 54), "test")
        .expect("wide powers of two remain exactly representable");
    let wide = (1_u64 << 53) + 1;
    let predictor_error = block_on(fitlm_builtin(
        None,
        poisoned_int_tensor(IntegerStorage::U64(vec![1, 2, wide]), 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor_value(
            vec![1.0, 2.0, 3.0],
            3,
            1,
        )]),
    ))
    .unwrap_err();
    assert!(predictor_error
        .message()
        .contains("exactly representable as double"));

    let response_error = block_on(fitlm_builtin(
        None,
        tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![poisoned_int_tensor(
            IntegerStorage::U64(vec![1, 2, wide]),
            3,
            1,
        )]),
    ))
    .unwrap_err();
    assert!(response_error
        .message()
        .contains("exactly representable as double"));

    let weights_error = block_on(fitlm_builtin(
        None,
        tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U64(vec![1, 2, wide]), 3, 1),
        ]),
    ))
    .unwrap_err();
    assert!(weights_error
        .message()
        .contains("exactly representable as double"));

    let selector_error = block_on(fitlm_builtin(
        None,
        tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            Value::from("Exclude"),
            poisoned_int_tensor(IntegerStorage::U64(vec![wide]), 1, 1),
        ]),
    ))
    .unwrap_err();
    assert!(selector_error.message().contains("exceeds observations"));
    assert!(!selector_error
        .message()
        .contains("exactly representable as double"));
}

#[test]
fn fitlm_strict_mode_gates_resident_fallback_before_gather() {
    use crate::builtins::common::{gpu_helpers, test_support};

    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1])
            .expect("integer predictors");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let floating = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("predictors");
        let floating_handle =
            gpu_helpers::upload_tensor(provider, &floating).expect("floating upload");
        let resident_error = block_on(fitlm_builtin(
            None,
            Value::GpuTensor(floating_handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![tensor_value(
                vec![1.0, 2.0, 3.0],
                3,
                1,
            )]),
        ))
        .unwrap_err();
        assert_eq!(
            resident_error.identifier(),
            Some("RunMat:compatibility:FitlmResidentFallbackExtension")
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(&floating_handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        let resident_option_error = block_on(fitlm_builtin(
            None,
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
                Value::from("Weights"),
                Value::GpuTensor(floating_handle.clone()),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            resident_option_error.identifier(),
            Some("RunMat:compatibility:FitlmResidentFallbackExtension")
        );

        let error = block_on(fitlm_builtin(
            None,
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![tensor_value(
                vec![1.0, 2.0, 3.0],
                3,
                1,
            )]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FitlmIntegerPredictorExtension")
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(&handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
    });
}
