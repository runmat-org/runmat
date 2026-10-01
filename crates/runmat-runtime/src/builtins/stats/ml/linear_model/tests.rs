use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn all_fitlm_integer_storages() -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![1, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 3]),
    ]
}

#[test]
fn scalar_option_parsers_read_all_integer_storage_variants() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let value = poisoned_int_tensor(storage, 1, 1);
        assert_eq!(numeric_scalar(&value, "Alpha").unwrap(), 1.0);
        assert!(bool_scalar(&value, "RobustOpts").unwrap());
        assert!(
            matches!(exclude_spec(&value), Ok(ExcludeSpec::Indices(indices)) if indices == vec![1])
        );
    }
}

fn object(value: Value) -> ObjectInstance {
    match value {
        Value::Object(object) => object,
        other => panic!("expected object, got {other:?}"),
    }
}

fn tensor(value: &Value) -> &Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn fitlm_matrix_fits_coefficients_and_predicts() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(x, vec![y])).unwrap();
    let model_object = object(model.clone());
    assert_eq!(
        model_object.properties.get("NumObservations"),
        Some(&Value::Num(4.0))
    );
    let coeffs = match model_object.properties.get("Coefficients").unwrap() {
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
    assert_eq!(ypred.shape, vec![2, 1]);
    assert!((ypred.materialize_f64()[0] - 9.0).abs() < 1.0e-10);
    assert!((ypred.materialize_f64()[1] - 11.0).abs() < 1.0e-10);
}

#[test]
fn registered_predict_preserves_statistical_output_overloads_without_ambient_arity() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(x, vec![y])).unwrap();
    let args = [model, tensor_value(vec![4.0, 5.0], 2, 1)];
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("predict");
    assert_eq!(bindings.len(), 1);
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _ambient_outputs = crate::output_count::push_output_count(Some(3));
    let predicted = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime.clone()), Some(1)),
    ))
    .unwrap();
    let Value::OutputList(predicted) = predicted else {
        panic!("one canonical statistical output expected")
    };
    let values = tensor(&predicted[0]);
    assert_eq!(values.shape, vec![2, 1]);
    assert!((values.materialize_f64()[0] - 9.0).abs() < 1.0e-10);

    let predicted = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime), Some(2)),
    ))
    .unwrap();
    assert!(matches!(predicted, Value::OutputList(values) if values.len() == 2));
}

#[test]
fn fitlm_and_predict_read_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let model = block_on(fitlm_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 99]), 4, 1),
        vec![
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 999]), 4, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1]), 4, 1),
            Value::from("Exclude"),
            poisoned_int_tensor(IntegerStorage::U8(vec![4]), 1, 1),
            Value::from("Intercept"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
        ],
    ))
    .unwrap();
    let predicted = block_on(predict_builtin(
        model,
        poisoned_int_tensor(IntegerStorage::I16(vec![3, 4]), 2, 1),
        Vec::new(),
    ))
    .unwrap();
    let ypred = tensor(&predicted);
    assert_eq!(ypred.shape, vec![2, 1]);
    assert!((ypred.materialize_f64()[0] - 7.0).abs() < 1.0e-10);
    assert!((ypred.materialize_f64()[1] - 9.0).abs() < 1.0e-10);
}

#[test]
fn fitlm_runmat_mode_admits_all_integer_classes_by_role() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_fitlm_integer_storages() {
        let model = block_on(fitlm_builtin(
            poisoned_int_tensor(storage.clone(), 3, 1),
            vec![
                poisoned_int_tensor(storage.clone(), 3, 1),
                Value::from("Weights"),
                poisoned_int_tensor(storage, 3, 1),
                Value::from("Exclude"),
                poisoned_int_tensor(IntegerStorage::U8(Vec::new()), 0, 1),
                Value::from("Intercept"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            ],
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
            block_on(fitlm_builtin(integer(), vec![y()])).unwrap_err(),
            "RunMat:compatibility:FitlmIntegerPredictorExtension",
        ),
        (
            block_on(fitlm_builtin(x(), vec![integer()])).unwrap_err(),
            "RunMat:compatibility:FitlmIntegerResponseExtension",
        ),
        (
            block_on(fitlm_builtin(
                x(),
                vec![y(), Value::from("Weights"), integer()],
            ))
            .unwrap_err(),
            "RunMat:compatibility:FitlmIntegerWeightsExtension",
        ),
        (
            block_on(fitlm_builtin(
                x(),
                vec![
                    y(),
                    Value::from("Exclude"),
                    poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                ],
            ))
            .unwrap_err(),
            "RunMat:compatibility:FitlmIntegerSelectorExtension",
        ),
        (
            block_on(fitlm_builtin(
                x(),
                vec![
                    y(),
                    Value::from("Intercept"),
                    poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                ],
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
        poisoned_int_tensor(IntegerStorage::U64(vec![1, 2, wide]), 3, 1),
        vec![tensor_value(vec![1.0, 2.0, 3.0], 3, 1)],
    ))
    .unwrap_err();
    assert!(predictor_error
        .message()
        .contains("exactly representable as double"));

    let response_error = block_on(fitlm_builtin(
        tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
        vec![poisoned_int_tensor(
            IntegerStorage::U64(vec![1, 2, wide]),
            3,
            1,
        )],
    ))
    .unwrap_err();
    assert!(response_error
        .message()
        .contains("exactly representable as double"));

    let weights_error = block_on(fitlm_builtin(
        tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
        vec![
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U64(vec![1, 2, wide]), 3, 1),
        ],
    ))
    .unwrap_err();
    assert!(weights_error
        .message()
        .contains("exactly representable as double"));

    let selector_error = block_on(fitlm_builtin(
        tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
        vec![
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            Value::from("Exclude"),
            poisoned_int_tensor(IntegerStorage::U64(vec![wide]), 1, 1),
        ],
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
            Value::GpuTensor(floating_handle.clone()),
            vec![tensor_value(vec![1.0, 2.0, 3.0], 3, 1)],
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
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            vec![
                tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
                Value::from("Weights"),
                Value::GpuTensor(floating_handle.clone()),
            ],
        ))
        .unwrap_err();
        assert_eq!(
            resident_option_error.identifier(),
            Some("RunMat:compatibility:FitlmResidentFallbackExtension")
        );

        let error = block_on(fitlm_builtin(
            Value::GpuTensor(handle.clone()),
            vec![tensor_value(vec![1.0, 2.0, 3.0], 3, 1)],
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

#[test]
fn fitlm_table_formula_uses_named_predictors() {
    let table = table_from_columns(
        vec!["A".into(), "B".into(), "Y".into()],
        vec![
            tensor_value(vec![1.0, 2.0, 3.0, 4.0], 4, 1),
            tensor_value(vec![0.0, 1.0, 0.0, 1.0], 4, 1),
            tensor_value(vec![3.0, 7.0, 9.0, 13.0], 4, 1),
        ],
    )
    .unwrap();
    let model = block_on(fitlm_builtin(
        table,
        vec![Value::String("Y ~ A + B".into())],
    ))
    .unwrap();
    let object = object(model);
    assert_eq!(
        object.properties.get("Formula"),
        Some(&Value::String("Y ~ A + B".into()))
    );
    let names = match object.properties.get("PredictorNames").unwrap() {
        Value::StringArray(array) => array.data.clone(),
        other => panic!("names {other:?}"),
    };
    assert_eq!(names, vec!["A", "B"]);
}

#[test]
fn fitlm_quadratic_adds_square_term() {
    let x = tensor_value(vec![-1.0, 0.0, 1.0, 2.0], 4, 1);
    let y = tensor_value(vec![3.0, 1.0, 3.0, 9.0], 4, 1);
    let model =
        object(block_on(fitlm_builtin(x, vec![y, Value::String("quadratic".into())])).unwrap());
    let coeffs = match model.properties.get("Coefficients").unwrap() {
        Value::Object(object) => table_variables(object).unwrap(),
        other => panic!("coefficients {other:?}"),
    };
    let estimates = tensor(coeffs.fields.get("Estimate").unwrap());
    assert_eq!(estimates.shape, vec![3, 1]);
    assert!((estimates.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!(estimates.materialize_f64()[1].abs() < 1.0e-10);
    assert!((estimates.materialize_f64()[2] - 2.0).abs() < 1.0e-10);
}

#[test]
fn fitlm_omits_nan_rows() {
    let x = tensor_value(vec![0.0, 1.0, f64::NAN, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = object(block_on(fitlm_builtin(x, vec![y])).unwrap());
    assert_eq!(
        model.properties.get("NumObservations"),
        Some(&Value::Num(3.0))
    );
}

#[test]
fn predict_confidence_intervals_are_column_major() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.1, 4.9, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(x, vec![y])).unwrap();
    let output_count = crate::output_count::push_output_count(Some(2));
    let predicted = block_on(predict_builtin(
        model,
        tensor_value(vec![4.0, 5.0], 2, 1),
        Vec::new(),
    ))
    .unwrap();
    drop(output_count);
    let yci = match predicted {
        Value::OutputList(list) => tensor(&list[1]).clone(),
        other => panic!("expected tensor or output list, got {other:?}"),
    };
    assert_eq!(yci.shape, vec![2, 2]);
    assert!(yci.materialize_f64()[0] < yci.materialize_f64()[1]);
    assert!(yci.materialize_f64()[2] < yci.materialize_f64()[3]);
    assert!(yci.materialize_f64()[0] < yci.materialize_f64()[2]);
    assert!(yci.materialize_f64()[1] < yci.materialize_f64()[3]);
}

#[test]
fn fitlm_formula_no_intercept_and_intercept_only_are_supported() {
    let table = table_from_columns(
        vec!["A".into(), "Y".into()],
        vec![
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            tensor_value(vec![2.0, 4.0, 6.0], 3, 1),
        ],
    )
    .unwrap();
    let no_intercept = object(
        block_on(fitlm_builtin(
            table.clone(),
            vec![Value::String("Y ~ A - 1".into())],
        ))
        .unwrap(),
    );
    let coeffs = match no_intercept.properties.get("Coefficients").unwrap() {
        Value::Object(object) => table_variables(object).unwrap(),
        other => panic!("coefficients {other:?}"),
    };
    let estimates = tensor(coeffs.fields.get("Estimate").unwrap());
    assert_eq!(estimates.shape, vec![1, 1]);
    assert!((estimates.materialize_f64()[0] - 2.0).abs() < 1.0e-10);

    let intercept_only =
        object(block_on(fitlm_builtin(table, vec![Value::String("Y ~ 1".into())])).unwrap());
    let coeffs = match intercept_only.properties.get("Coefficients").unwrap() {
        Value::Object(object) => table_variables(object).unwrap(),
        other => panic!("coefficients {other:?}"),
    };
    let estimates = tensor(coeffs.fields.get("Estimate").unwrap());
    assert_eq!(estimates.shape, vec![1, 1]);
    assert!((estimates.materialize_f64()[0] - 4.0).abs() < 1.0e-10);
}

#[test]
fn intercept_only_table_model_predicts_one_value_per_table_row() {
    let train = table_from_columns(
        vec!["A".into(), "Y".into()],
        vec![
            tensor_value(vec![1.0, 2.0, 3.0], 3, 1),
            tensor_value(vec![2.0, 4.0, 6.0], 3, 1),
        ],
    )
    .unwrap();
    let model = block_on(fitlm_builtin(train, vec![Value::String("Y ~ 1".into())])).unwrap();
    let xnew = table_from_columns(
        vec!["A".into(), "Y".into()],
        vec![
            tensor_value(vec![10.0, 20.0], 2, 1),
            tensor_value(vec![0.0, 0.0], 2, 1),
        ],
    )
    .unwrap();
    let predicted = block_on(predict_builtin(model, xnew, Vec::new())).unwrap();
    let yhat = tensor(&predicted);
    assert_eq!(yhat.shape, vec![2, 1]);
    assert_eq!(yhat.materialize_f64(), vec![4.0, 4.0]);
}

#[test]
fn predict_rejects_unsupported_simultaneous_intervals() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(x, vec![y])).unwrap();
    let err = block_on(predict_builtin(
        model,
        tensor_value(vec![4.0], 1, 1),
        vec![Value::from("Simultaneous"), Value::Bool(true)],
    ))
    .unwrap_err();
    assert!(err.message.contains("Simultaneous=true"));
}

#[test]
fn fitlm_matrix_rejects_table_only_options_and_bad_varnames() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let err = block_on(fitlm_builtin(
        x.clone(),
        vec![
            y.clone(),
            Value::from("ResponseVar"),
            Value::String("Y".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("ResponseVar"));

    let err = block_on(fitlm_builtin(
        x.clone(),
        vec![
            y.clone(),
            Value::from("PredictorVars"),
            Value::String("X".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("PredictorVars"));

    let err = block_on(fitlm_builtin(
        x,
        vec![
            y,
            Value::from("VarNames"),
            Value::StringArray(StringArray {
                data: vec!["only_x".into()],
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            }),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("VarNames"));
}

#[test]
fn coefficients_table_exposes_row_names() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = object(block_on(fitlm_builtin(x, vec![y])).unwrap());
    let coeffs = match model.properties.get("Coefficients").unwrap() {
        Value::Object(object) => object,
        other => panic!("coefficients {other:?}"),
    };
    let props = match coeffs.properties.get("Properties").unwrap() {
        Value::Struct(props) => props,
        other => panic!("properties {other:?}"),
    };
    match props.fields.get("RowNames").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data, vec!["(Intercept)", "x1"]);
            assert_eq!(array.shape, vec![2, 1]);
        }
        other => panic!("row names {other:?}"),
    }
}

#[test]
fn fitlm_accepts_exclude_indices_and_table_weight_variable() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![100.0, 3.0, 5.0, 7.0], 4, 1);
    let model = object(
        block_on(fitlm_builtin(
            x,
            vec![y, Value::from("Exclude"), tensor_value(vec![1.0], 1, 1)],
        ))
        .unwrap(),
    );
    assert_eq!(
        model.properties.get("NumObservations"),
        Some(&Value::Num(3.0))
    );

    let table = table_from_columns(
        vec!["A".into(), "Y".into(), "W".into()],
        vec![
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1),
            tensor_value(vec![1.0, 1.0, 1.0, 1.0], 4, 1),
        ],
    )
    .unwrap();
    let weighted = object(
        block_on(fitlm_builtin(
            table,
            vec![
                Value::String("Y ~ A".into()),
                Value::from("Weights"),
                Value::String("W".into()),
            ],
        ))
        .unwrap(),
    );
    assert_eq!(
        weighted.properties.get("NumPredictors"),
        Some(&Value::Num(1.0))
    );
}
