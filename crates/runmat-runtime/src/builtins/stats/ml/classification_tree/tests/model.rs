use super::*;

#[test]
fn fitctree_retains_rows_with_partially_missing_predictors() {
    let model = block_on(fitctree_builtin(
        None,
        tensor_value(vec![0.0, 1.0, 2.0, 3.0, f64::NAN, 10.0, 11.0, 12.0], 4, 2),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let Value::Object(object) = model else {
        panic!("expected model");
    };
    assert_eq!(
        object.properties().get("NumObservations"),
        Some(&Value::Num(4.0))
    );
}

#[test]
fn fitctree_table_response_name_predicts_string_labels() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "Y".into()],
        vec![
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "low".into(), "high".into(), "high".into()],
                    vec![4, 1],
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let model = block_on(fitctree_builtin(
        None,
        table.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Y".into()),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        model,
        table,
        Vec::new(),
    ))
    .unwrap();
    let Value::StringArray(labels) = predicted else {
        panic!("expected string labels");
    };
    assert_eq!(labels.shape, vec![4, 1]);
    assert_eq!(labels.data()[0], "low");
    assert_eq!(labels.data()[3], "high");
}

#[test]
fn fitctree_respects_class_names_and_depth_controls() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![2.0, 2.0, 5.0, 5.0], 4, 1);
    let model = object(
        block_on(fitctree_builtin(
            None,
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![
                y,
                Value::from("ClassNames"),
                tensor_value(vec![5.0, 2.0], 2, 1),
                Value::from("MaxNumSplits"),
                Value::Num(0.0),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(model.properties().get("NumSplits"), Some(&Value::Num(0.0)));
    let classes = tensor(model.properties().get("ClassNames").unwrap());
    assert_eq!(classes.materialize_f64(), vec![5.0, 2.0]);
}

#[test]
fn fitctree_omits_nan_responses_and_requires_two_observed_classes() {
    let model = object(
        block_on(fitctree_builtin(
            None,
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor_value(vec![0.0, f64::NAN, 1.0, 1.0], 4, 1),
                Value::from("MinParentSize"),
                Value::Num(2.0),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(
        model.properties().get("NumObservations"),
        Some(&Value::Num(3.0))
    );
    assert_eq!(
        tensor(model.properties().get("ClassNames").unwrap()).materialize_f64(),
        vec![0.0, 1.0]
    );

    let err = block_on(fitctree_builtin(
        None,
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![0.0, f64::NAN, f64::NAN, 0.0], 4, 1),
            Value::from("ClassNames"),
            tensor_value(vec![0.0, 1.0], 2, 1),
        ]),
    ))
    .unwrap_err();
    assert!(err.message().contains("at least two observed classes"));
}

#[test]
fn fitctree_excludes_table_weight_variable_from_default_predictors() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "W".into(), "Y".into()],
        vec![
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            tensor_value(vec![1.0, 2.0, 1.0, 2.0], 4, 1),
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
        ],
    )
    .unwrap();
    let model = object(
        block_on(fitctree_builtin(
            None,
            table,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Y"),
                Value::from("Weights"),
                Value::from("W"),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(
        model.properties().get("NumPredictors"),
        Some(&Value::Num(1.0))
    );
    let Value::StringArray(names) = model.properties().get("PredictorNames").unwrap() else {
        panic!("expected predictor names");
    };
    assert_eq!(names.data(), vec!["A"]);
}

#[test]
fn fitctree_predict_rejects_too_many_outputs() {
    let model = block_on(fitctree_builtin(
        None,
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let output_count = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let err = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        model,
        tensor_value(vec![0.5, 2.5], 2, 1),
        Vec::new(),
    ))
    .unwrap_err();
    drop(output_count);
    assert!(err.message().contains("too many output"));
}

#[test]
fn fitctree_rejects_unsupported_cross_validation_options() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
    let err = block_on(fitctree_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            y,
            Value::from("CrossVal"),
            Value::String("on".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("not supported"));
}
