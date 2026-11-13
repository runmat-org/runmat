use super::*;

#[tokio::test]
async fn fitclinear_logistic_returns_fitinfo_and_probability_scores() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = fitclinear_builtin(
        None,
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::String("Learner".into()),
            Value::String("logistic".into()),
            Value::String("Lambda".into()),
            Value::Num(0.0),
        ]),
    )
    .await
    .unwrap();
    let (object, info) = match value {
        Value::OutputList(outputs) => {
            let mut outputs = outputs
                .try_into_owned()
                .expect("unique outputs")
                .into_iter();
            let object = match outputs.next().unwrap() {
                Value::Object(object) => object,
                other => panic!("expected object, got {other:?}"),
            };
            let info = outputs.next().unwrap();
            (object, info)
        }
        other => panic!("expected outputs, got {other:?}"),
    };
    assert!(matches!(info, Value::Struct(_)));
    let outputs = predict_classification_linear_object(
        object,
        tensor(vec![0.2, 2.8], vec![2, 1]),
        crate::call::FunctionArgumentView::from_slice(&[]),
    )
    .unwrap();
    match &outputs[1] {
        Value::Tensor(score) => {
            assert_eq!(score.shape(), &[2, 2]);
            assert!((score.materialize_f64()[0] + score.materialize_f64()[2] - 1.0).abs() < 1.0e-8);
        }
        other => panic!("expected score, got {other:?}"),
    }
}

#[tokio::test]
async fn fitclinear_observations_in_columns_and_text_classnames() {
    let model = fitclinear_builtin(
        None,
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "low".into(), "high".into(), "high".into()],
                    vec![4, 1],
                )
                .unwrap(),
            ),
            Value::String("ObservationsIn".into()),
            Value::String("columns".into()),
            Value::String("ClassNames".into()),
            Value::StringArray(
                StringArray::new(vec!["low".into(), "high".into()], vec![2, 1]).unwrap(),
            ),
        ]),
    )
    .await
    .unwrap();
    let outputs = predict_classification_linear_object(
        match model {
            Value::Object(object) => object,
            _ => panic!("expected object"),
        },
        tensor(vec![0.2, 2.8], vec![1, 2]),
        crate::call::FunctionArgumentView::from_slice(&[
            Value::String("ObservationsIn".into()),
            Value::String("columns".into()),
        ]),
    )
    .unwrap();
    match &outputs[0] {
        Value::StringArray(labels) => assert_eq!(labels.data(), vec!["low", "high"]),
        other => panic!("expected string labels, got {other:?}"),
    }
}

#[tokio::test]
async fn fitclinear_rejects_multiclass_response() {
    let err = fitclinear_builtin(
        None,
        tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![0.0, 1.0, 2.0],
            vec![3, 1],
        )]),
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("exactly two classes"));
}

#[tokio::test]
async fn fitclinear_table_external_y_honors_response_name() {
    let table = table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
            tensor(vec![1.0, 1.0, 1.0, 1.0], vec![4, 1]),
        ],
    )
    .unwrap();
    let model = fitclinear_builtin(
        None,
        table,
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::String("ResponseName".into()),
            Value::String("Outcome".into()),
        ]),
    )
    .await
    .unwrap();
    match model {
        Value::Object(object) => {
            assert_eq!(
                object.properties().get("ResponseName"),
                Some(&Value::String("Outcome".into()))
            );
        }
        other => panic!("expected object, got {other:?}"),
    }
}
