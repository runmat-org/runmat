use super::*;

#[test]
fn fitclinear_header_keeps_typed_options_and_ignored_payload_policy() {
    let first = tensor(vec![0.0, 1.0], vec![2, 1]);
    let values = vec![
        Value::Num(0.0),
        Value::from("IterationLimit"),
        Value::Num(5.0),
        Value::from("BatchLimit"),
        Value::Num(20.0),
    ];
    let header = HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&values),
    )
    .unwrap()
    .validate(values.len())
    .unwrap();
    assert_eq!(header.option_start, 1);
    assert_eq!(
        header.options[0].name.identity,
        FitclinearOption::IterationLimit
    );
    assert_eq!(
        header.options[1].name.identity,
        FitclinearOption::BatchLimit
    );
    assert!(header.options[1].name.identity.ignored());

    let invalid = vec![Value::Num(0.0), Value::from("Cost"), Value::Num(1.0)];
    assert!(HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&invalid)
    )
    .unwrap()
    .validate(invalid.len())
    .is_err());
}

#[test]
fn fitclinear_rejects_known_rank_before_materialization() {
    let first = tensor(vec![0.0; 8], vec![2, 2, 2]);
    assert!(preflight_matrix_rank(&first).is_err());
}

#[test]
fn typed_fitclinear_actions_preserve_ignored_payload_and_response_orientation() {
    let first = tensor(vec![0.0; 8], vec![2, 4]);
    let values = vec![
        tensor(vec![0.0, 1.0, 0.0, 1.0], vec![4, 1]),
        Value::from("ObservationsIn"),
        Value::from("columns"),
        Value::from("BatchLimit"),
        Value::from("uninspected payload"),
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
        prepared.options[0].predecoded.as_ref(),
        Some(FitclinearAction::ObservationsIn(ObservationsIn::Columns))
    ));
    assert!(matches!(
        prepared.options[1].predecoded.as_ref(),
        Some(FitclinearAction::Ignored)
    ));
    preflight_response_shape(
        &first,
        crate::call::FunctionArgumentView::from_slice(&values),
        &prepared,
    )
    .unwrap();
    let wrong_y = vec![
        tensor(vec![0.0, 1.0], vec![2, 1]),
        values[1].clone(),
        values[2].clone(),
    ];
    let wrong = HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&wrong_y),
    )
    .unwrap()
    .validate(wrong_y.len())
    .unwrap()
    .preflight_values(crate::call::FunctionArgumentView::from_slice(&wrong_y))
    .unwrap();
    assert!(preflight_response_shape(
        &first,
        crate::call::FunctionArgumentView::from_slice(&wrong_y),
        &wrong
    )
    .expect_err("known response mismatch")
    .message()
    .contains("length(Y)"));
    let invalid = vec![
        values[0].clone(),
        Value::from("ObservationsIn"),
        Value::from("sideways"),
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
    .expect("invalid orientation")
    .message()
    .contains("ObservationsIn"));
}

#[test]
fn iteration_limit_rejects_the_first_value_beyond_platform_usize() {
    let first_out_of_range = (1u128 << usize::BITS) as f64;
    assert!(positive_integer(&Value::Num(first_out_of_range), "IterationLimit").is_err());
    assert_eq!(
        positive_integer(&Value::Num(1.0), "IterationLimit").unwrap(),
        1
    );
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
        let value = poisoned_int_tensor(storage, vec![1, 1]);
        assert_eq!(numeric_scalar(&value, "Bias").unwrap(), 1.0);
        assert!(logical_scalar(&value, "FitBias").unwrap());
        assert_eq!(positive_integer(&value, "BatchLimit").unwrap(), 1);
    }
}

#[tokio::test]
async fn fitclinear_svm_predicts_binary_numeric_labels() {
    let model = fitclinear_builtin(
        None,
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::String("Lambda".into()),
            Value::Num(0.0),
        ]),
    )
    .await
    .unwrap();
    let outputs = predict_classification_linear_object(
        match model {
            Value::Object(object) => object,
            _ => panic!("expected object"),
        },
        tensor(vec![0.2, 2.8], vec![2, 1]),
        crate::call::FunctionArgumentView::from_slice(&[]),
    )
    .unwrap();
    match &outputs[0] {
        Value::Tensor(labels) => assert_eq!(labels.materialize_f64(), vec![0.0, 1.0]),
        other => panic!("expected labels, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(score) => assert_eq!(score.shape(), &[2, 2]),
        other => panic!("expected score, got {other:?}"),
    }
}

#[tokio::test]
async fn fitclinear_and_predict_read_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let model = fitclinear_builtin(
        None,
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            poisoned_int_tensor(IntegerStorage::U8(vec![0, 0, 1, 1]), vec![4, 1]),
            Value::String("Lambda".into()),
            Value::Num(0.0),
            Value::String("FitBias".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            Value::String("IterationLimit".into()),
            poisoned_int_tensor(IntegerStorage::U16(vec![25]), vec![1, 1]),
            Value::String("Bias".into()),
            poisoned_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
        ]),
    )
    .await
    .unwrap();
    let outputs = predict_classification_linear_object(
        match model {
            Value::Object(object) => object,
            _ => panic!("expected object"),
        },
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 3]), vec![2, 1]),
        crate::call::FunctionArgumentView::from_slice(&[]),
    )
    .unwrap();
    match &outputs[0] {
        Value::Tensor(labels) => assert_eq!(labels.materialize_f64(), vec![0.0, 1.0]),
        other => panic!("expected labels, got {other:?}"),
    }
}
