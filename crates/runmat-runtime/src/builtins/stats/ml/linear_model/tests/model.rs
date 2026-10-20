use super::*;

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
        None,
        table,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "Y ~ A + B".into(),
        )]),
    ))
    .unwrap();
    let object = object(model);
    assert_eq!(
        object.properties().get("Formula"),
        Some(&Value::String("Y ~ A + B".into()))
    );
    let names = match object.properties().get("PredictorNames").unwrap() {
        Value::StringArray(array) => array.data(),
        other => panic!("names {other:?}"),
    };
    assert_eq!(names, vec!["A", "B"]);
}

#[test]
fn fitlm_quadratic_adds_square_term() {
    let x = tensor_value(vec![-1.0, 0.0, 1.0, 2.0], 4, 1);
    let y = tensor_value(vec![3.0, 1.0, 3.0, 9.0], 4, 1);
    let model = object(
        block_on(fitlm_builtin(
            None,
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![
                y,
                Value::String("quadratic".into()),
            ]),
        ))
        .unwrap(),
    );
    let coeffs = match model.properties().get("Coefficients").unwrap() {
        Value::Object(object) => table_variables(object).unwrap(),
        other => panic!("coefficients {other:?}"),
    };
    let estimates = tensor(coeffs.fields.get("Estimate").unwrap());
    assert_eq!(estimates.shape(), [3, 1]);
    assert!((estimates.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!(estimates.materialize_f64()[1].abs() < 1.0e-10);
    assert!((estimates.materialize_f64()[2] - 2.0).abs() < 1.0e-10);
}

#[test]
fn fitlm_omits_nan_rows() {
    let x = tensor_value(vec![0.0, 1.0, f64::NAN, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = object(
        block_on(fitlm_builtin(
            None,
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![y]),
        ))
        .unwrap(),
    );
    assert_eq!(
        model.properties().get("NumObservations"),
        Some(&Value::Num(3.0))
    );
}

#[test]
fn predict_confidence_intervals_are_column_major() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.1, 4.9, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![y]),
    ))
    .unwrap();
    let output_count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
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
    assert_eq!(yci.shape(), [2, 2]);
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
            None,
            table.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "Y ~ A - 1".into(),
            )]),
        ))
        .unwrap(),
    );
    let coeffs = match no_intercept.properties().get("Coefficients").unwrap() {
        Value::Object(object) => table_variables(object).unwrap(),
        other => panic!("coefficients {other:?}"),
    };
    let estimates = tensor(coeffs.fields.get("Estimate").unwrap());
    assert_eq!(estimates.shape(), [1, 1]);
    assert!((estimates.materialize_f64()[0] - 2.0).abs() < 1.0e-10);

    let intercept_only = object(
        block_on(fitlm_builtin(
            None,
            table,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "Y ~ 1".into(),
            )]),
        ))
        .unwrap(),
    );
    let coeffs = match intercept_only.properties().get("Coefficients").unwrap() {
        Value::Object(object) => table_variables(object).unwrap(),
        other => panic!("coefficients {other:?}"),
    };
    let estimates = tensor(coeffs.fields.get("Estimate").unwrap());
    assert_eq!(estimates.shape(), [1, 1]);
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
    let model = block_on(fitlm_builtin(
        None,
        train,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "Y ~ 1".into(),
        )]),
    ))
    .unwrap();
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
    assert_eq!(yhat.shape(), [2, 1]);
    assert_eq!(yhat.materialize_f64(), vec![4.0, 4.0]);
}

#[test]
fn predict_rejects_unsupported_simultaneous_intervals() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = block_on(fitlm_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![y]),
    ))
    .unwrap();
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
        None,
        x.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            y.clone(),
            Value::from("ResponseVar"),
            Value::String("Y".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("ResponseVar"));

    let err = block_on(fitlm_builtin(
        None,
        x.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            y.clone(),
            Value::from("PredictorVars"),
            Value::String("X".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("PredictorVars"));

    let err = block_on(fitlm_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            y,
            Value::from("VarNames"),
            Value::StringArray(StringArray::new(vec!["only_x".into()], vec![1, 1]).unwrap()),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("VarNames"));
}

#[test]
fn coefficients_table_exposes_row_names() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let model = object(
        block_on(fitlm_builtin(
            None,
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![y]),
        ))
        .unwrap(),
    );
    let coeffs = match model.properties().get("Coefficients").unwrap() {
        Value::Object(object) => object,
        other => panic!("coefficients {other:?}"),
    };
    let props = match coeffs.properties().get("Properties").unwrap() {
        Value::Struct(props) => props,
        other => panic!("properties {other:?}"),
    };
    match props.fields.get("RowNames").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data(), vec!["(Intercept)", "x1"]);
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
            None,
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![
                y,
                Value::from("Exclude"),
                tensor_value(vec![1.0], 1, 1),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(
        model.properties().get("NumObservations"),
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
            None,
            table,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Y ~ A".into()),
                Value::from("Weights"),
                Value::String("W".into()),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(
        weighted.properties().get("NumPredictors"),
        Some(&Value::Num(1.0))
    );
}
