use super::*;
use runmat_builtins::{builtin_function_by_name, BuiltinIntegerOutputClassRule};

#[test]
fn classify_registers_independent_integer_extensions_and_capabilities() {
    let builtin = runmat_builtins::builtin_catalog_primary_entry_by_name(CLASSIFY_NAME)
        .expect("canonical classify identity");
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(CLASSIFY_NAME);
    assert_eq!(bindings.len(), 1, "one canonical implementation binding");
    assert_eq!(bindings[0].identity.builtin, builtin.identity);
    assert_eq!(bindings[0].identity.variant, "default");
    assert!(
        builtin_function_by_name(CLASSIFY_NAME).is_none(),
        "no legacy authority"
    );
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
        assert_eq!(posterior.shape(), &[2, 2]);
        let posterior = posterior.as_f64_slice().expect("double posterior");
        assert!(posterior[0] > posterior[2]);
        assert!(posterior[3] > posterior[1]);
        let Value::StructArray(coeff) = &values[4] else {
            panic!("coeff");
        };
        assert_eq!(coeff.shape(), &[2, 2]);
        assert!(coeff
            .get_linear(1)
            .expect("pairwise coefficient")
            .fields
            .contains_key("linear"));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn classify_coefficients_preserve_three_class_column_major_pair_orientation() {
    with_outputs(5, || {
        let values = classify(
            tensor(vec![0.2, 2.2, 4.2], vec![3, 1]),
            tensor(vec![0.0, 0.4, 2.0, 2.4, 4.0, 4.4], vec![6, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![6, 1]),
            Vec::new(),
        );
        let Value::StructArray(coeff) = &values[4] else {
            panic!("coefficients must be a structure array");
        };
        assert_eq!(coeff.shape(), &[3, 3]);
        for col in 0..3 {
            for row in 0..3 {
                let element = coeff.get_linear(row + col * 3).expect("pairwise entry");
                let expected_name1 = tensor(vec![(row + 1) as f64], vec![1, 1]);
                let expected_name2 = tensor(vec![(col + 1) as f64], vec![1, 1]);
                assert_eq!(element.fields.get("name1"), Some(&expected_name1));
                assert_eq!(element.fields.get("name2"), Some(&expected_name2));
                assert!(element.fields.contains_key("const"));
                assert!(element.fields.contains_key("linear"));
                assert!(!element.fields.contains_key("quadratic"));
            }
        }
        let forward = coeff.get_linear(3).expect("class 1 versus 2");
        let reverse = coeff.get_linear(1).expect("class 2 versus 1");
        let (Some(Value::Num(forward_const)), Some(Value::Num(reverse_const))) =
            (forward.fields.get("const"), reverse.fields.get("const"))
        else {
            panic!("pairwise constants");
        };
        // Each class has mean 0.2, 2.2, or 4.2 and unbiased variance 0.08.
        // The pooled linear boundary for class 1 versus 2 is 30 - 25*x;
        // the numerical core adds only its 1e-9 covariance regularizer.
        assert!((*forward_const - 30.0).abs() < 1e-6);
        assert!((*reverse_const + 30.0).abs() < 1e-6);
        assert!((*forward_const + *reverse_const).abs() < 1e-10);
        let (Some(Value::Tensor(forward_linear)), Some(Value::Tensor(reverse_linear))) =
            (forward.fields.get("linear"), reverse.fields.get("linear"))
        else {
            panic!("pairwise linear coefficients");
        };
        assert_eq!(forward_linear.shape(), &[1, 1]);
        assert_eq!(reverse_linear.shape(), &[1, 1]);
        assert!((forward_linear.as_f64_slice().unwrap()[0] + 25.0).abs() < 1e-6);
        assert!((reverse_linear.as_f64_slice().unwrap()[0] - 25.0).abs() < 1e-6);
        assert!(
            (forward_linear.as_f64_slice().unwrap()[0] + reverse_linear.as_f64_slice().unwrap()[0])
                .abs()
                < 1e-10
        );
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
    assert_eq!(posterior.posterior.as_ref().unwrap().shape(), &[2, 2]);
    assert!(posterior.logp.is_none());

    let complete = predict_rows(&model, &sample, true, true).unwrap();
    assert_eq!(complete.posterior.as_ref().unwrap().shape(), &[2, 2]);
    assert_eq!(complete.logp.as_ref().unwrap().shape(), &[2, 1]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn classification_matrix_dimensions_preserve_numerical_core_projection() {
    let rank_zero = Tensor::new(vec![1.0], Vec::new()).unwrap();
    let rank_one = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
    let matrix = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    assert_eq!(classification_matrix_dimensions(&rank_zero), (0, 0));
    assert_eq!(classification_matrix_dimensions(&rank_one), (1, 2));
    assert_eq!(classification_matrix_dimensions(&matrix), (2, 1));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn classify_rank_zero_and_rank_one_inputs_preserve_existing_late_geometry_rejection() {
    let training = || tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]);
    let groups = || tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]);
    for sample in [
        tensor(vec![0.2], Vec::new()),
        tensor(vec![0.2, 2.2], vec![2]),
    ] {
        let error = block_on(classify_builtin(
            sample,
            training(),
            groups(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("header and numerical matrix projections remain distinct");
        assert!(error
            .message()
            .contains("sample and training must have the same number of columns"));
    }
}

#[test]
fn classify_normalizes_large_finite_prior_weights_without_overflow() {
    let labels = labels_from_value(tensor(vec![1.0, 2.0], vec![2, 1]), "group").unwrap();
    let prior =
        prior::PriorSpec::prepare(Some(&tensor(vec![1.0e308, 1.0e308], vec![2, 1]))).unwrap();
    let probabilities = prior
        .resolve(&labels.labels, labels.kind, &[vec![0], vec![1]])
        .unwrap();
    assert_eq!(probabilities, vec![0.5, 0.5]);
}
