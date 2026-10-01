use super::*;
use runmat_builtins::{builtin_function_by_name, BuiltinIntegerOutputClassRule};

#[test]
fn classify_registers_independent_integer_extensions_and_capabilities() {
    let builtin = builtin_function_by_name(CLASSIFY_NAME).expect("classify builtin");
    assert_eq!(builtin.extensions.len(), 6);
    assert_eq!(builtin.integer_capabilities.len(), 4);
    assert!(builtin
        .integer_capabilities
        .iter()
        .all(|capability| capability.inputs[0].classes.len() == 8));
    assert_eq!(
        builtin.integer_capabilities[2].output_class,
        BuiltinIntegerOutputClassRule::PreserveInput
    );
}

#[test]
fn classify_linear_predicts_numeric_groups() {
    with_outputs(5, || {
        let values = classify(
            tensor(vec![0.2, 2.2], vec![2, 1]),
            tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
            Vec::new(),
        );
        let Value::Tensor(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.as_f64_slice(), Some(&[1.0, 2.0][..]));
        assert!(matches!(values[1], Value::Num(err) if err <= EPS));
        let Value::Tensor(posterior) = &values[2] else {
            panic!("posterior");
        };
        assert_eq!(posterior.shape, vec![2, 2]);
        let posterior = posterior.as_f64_slice().expect("double posterior");
        assert!(posterior[0] > posterior[2]);
        assert!(posterior[3] > posterior[1]);
        let Value::Cell(coeff) = &values[4] else {
            panic!("coeff");
        };
        assert_eq!(coeff.rows, 2);
        assert_eq!(coeff.cols, 2);
        assert!(matches!(&coeff.data[1], Value::Struct(st) if st.fields.contains_key("linear")));
    });
}

#[test]
fn classify_materializes_only_requested_prediction_outputs() {
    let (sample, training, group) = valid_floating_inputs();
    let sample = numeric_matrix(sample, "sample").unwrap();
    let training = numeric_matrix(training, "training").unwrap();
    let labels = labels_from_value(group, "group").unwrap();
    let model = prepare_model(
        &training,
        labels,
        DiscriminantType::Linear,
        prior::PriorSpec::Uniform,
    )
    .unwrap();

    let labels_only = predict_rows(&model, &sample, false, false).unwrap();
    assert_eq!(labels_only.labels, vec![0, 1]);
    assert!(labels_only.posterior.is_none());
    assert!(labels_only.logp.is_none());

    let linear_coefficients = pairwise_coeff(&model, 0, 1).unwrap();
    assert!(linear_coefficients.quadratic.is_none());

    let posterior = predict_rows(&model, &sample, true, false).unwrap();
    assert_eq!(posterior.posterior.as_ref().unwrap().shape, vec![2, 2]);
    assert!(posterior.logp.is_none());

    let complete = predict_rows(&model, &sample, true, true).unwrap();
    assert_eq!(complete.posterior.as_ref().unwrap().shape, vec![2, 2]);
    assert_eq!(complete.logp.as_ref().unwrap().shape, vec![2, 1]);
}
