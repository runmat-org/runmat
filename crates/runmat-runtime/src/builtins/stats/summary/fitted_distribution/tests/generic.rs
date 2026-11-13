use super::*;
use std::cell::Cell;

#[test]
fn fitted_gamma_inverse_polls_during_search() {
    let polls = Cell::new(0);
    let error = invert_positive(
        0.7,
        2.0,
        |value| value / (1.0 + value),
        || {
            polls.set(polls.get() + 1);
            if polls.get() > 3 {
                Err(invalid("cancelled by fixture"))
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    assert!(polls.get() > 3);
    assert_eq!(error.message(), "cancelled by fixture");
}

#[test]
fn malformed_calls_fail_at_host_visible_admission() {
    let wrong_class = Value::Object(ObjectInstance::new("UnrelatedClass"));
    assert!(admission::evaluation(
        FittedOperation::Density,
        &wrong_class,
        &Value::Num(0.0),
        crate::call::FunctionArgumentView::from_slice(&[])
    )
    .is_err());
    assert!(admission::evaluation(
        FittedOperation::Cumulative,
        &Value::String("Normal".into()),
        &Value::Num(0.0),
        crate::call::FunctionArgumentView::from_slice(&[
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Num(2.0)
        ]),
    )
    .is_err());
    assert!(admission::fit(
        &Value::Num(1.0),
        &Value::String("Normal".into()),
        crate::call::FunctionArgumentView::from_slice(&[
            Value::String("Unsupported".into()),
            Value::Num(1.0)
        ]),
    )
    .is_err());
    let gamma = DistributionSelection::from_value(
        FittedOperation::Sampling,
        &Value::String("Gamma".into()),
    )
    .unwrap();
    assert!(admission::random(
        &gamma,
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0)])
    )
    .is_err());
}

#[test]
fn malformed_fitted_object_errors_use_the_invoked_public_operation() {
    let malformed = Value::Object(ObjectInstance::new(
        runmat_types::standard::PROBABILITY_DISTRIBUTION.to_string(),
    ));
    let pdf = admission::evaluation(
        FittedOperation::Density,
        &malformed,
        &Value::Num(0.0),
        crate::call::FunctionArgumentView::from_slice(&[]),
    )
    .err()
    .unwrap();
    assert_eq!(pdf.identifier(), Some("RunMat:pdf:InvalidArgument"));
    let cdf = admission::evaluation(
        FittedOperation::Cumulative,
        &malformed,
        &Value::Num(0.0),
        crate::call::FunctionArgumentView::from_slice(&[]),
    )
    .err()
    .unwrap();
    assert_eq!(cdf.identifier(), Some("RunMat:cdf:InvalidArgument"));
    let random = DistributionSelection::from_value(FittedOperation::Sampling, &malformed)
        .err()
        .unwrap();
    assert_eq!(random.identifier(), Some("RunMat:random:InvalidArgument"));
}

#[test]
fn random_provider_error_context_preserves_the_original_identifier() {
    let provider_error = build_runtime_error("provider rejected output")
        .with_identifier("Provider:Rejected")
        .build();
    let contextual = with_fitted_context(RANDOM_NAME, provider_error);
    assert_eq!(contextual.identifier(), Some("Provider:Rejected"));
    assert_eq!(contextual.context.builtin.as_deref(), Some(RANDOM_NAME));
}

#[test]
fn fitted_preflight_carries_typed_family_and_frequency_roles() {
    let options = vec![Value::from("freq"), Value::Num(1.0)];
    let admitted = admission::fit(
        &Value::Num(2.0),
        &Value::from("norm"),
        crate::call::FunctionArgumentView::from_slice(&options),
    )
    .expect("public aliases decode at admission");
    assert_eq!(admitted.kind, DistributionKind::Normal);
    assert!(matches!(admitted.frequency, Some(Value::Num(value)) if *value == 1.0));

    let selected =
        DistributionSelection::from_value(FittedOperation::Sampling, &Value::from("poiss"))
            .expect("public family decodes once");
    assert!(matches!(
        &selected,
        DistributionSelection::Named(DistributionKind::Poisson)
    ));
    assert_eq!(selected.parameter_count(), 1);
}

#[test]
fn generic_pdf_cdf_random_name_overloads_execute() {
    let _rng_guard = crate::builtins::common::random::test_guard();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let density = block_on(pdf_builtin(
        Value::String("Normal".into()),
        mirrorless_int_tensor(IntegerStorage::I16(vec![0, 1]), vec![2, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            mirrorless_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
            mirrorless_int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
        ]),
    ))
    .unwrap();
    let Value::Tensor(density) = density else {
        panic!("expected vector density");
    };
    assert_eq!(density.shape(), &[2, 1]);
    assert!(
        (density.materialize_f64()[0] - distribution_math::standard_normal_pdf(0.0)).abs()
            < 1.0e-12
    );

    let probability = block_on(cdf_builtin(
        Value::String("Poisson".into()),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
    ))
    .unwrap();
    let Value::Num(probability) = probability else {
        panic!("expected scalar probability");
    };
    assert!(probability > 0.0 && probability < 1.0);

    let samples = block_on(random_builtin(
        Value::String("Weibull".into()),
        vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(2.0),
            Value::Num(3.0),
        ],
    ))
    .unwrap();
    match samples {
        Value::Tensor(tensor) => assert_eq!(tensor.shape(), &[2, 3]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fitted_evaluators_preserve_rank_zero_and_rank_one_input_geometry() {
    let fitted = block_on(fitdist_builtin(
        vec_tensor(&[-1.0, 0.0, 1.0]),
        Value::from("Normal"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fitted normal distribution");

    for (shape, points, probabilities) in [
        (Vec::new(), vec![0.0], vec![0.5]),
        (vec![2], vec![-0.5, 0.5], vec![0.25, 0.75]),
    ] {
        let results = [
            block_on(pdf_builtin(
                fitted.clone(),
                Value::Tensor(Tensor::new(points.clone(), shape.clone()).unwrap()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
            block_on(cdf_builtin(
                fitted.clone(),
                Value::Tensor(Tensor::new(points, shape.clone()).unwrap()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
            block_on(
                crate::builtins::stats::summary::distributions::icdf::icdf_builtin(
                    fitted.clone(),
                    Value::Tensor(Tensor::new(probabilities, shape.clone()).unwrap()),
                    Vec::new(),
                ),
            ),
        ];

        for result in results {
            let result = result.expect("fitted evaluator");
            if shape.is_empty() {
                assert!(matches!(result, Value::Num(value) if value.is_finite()));
            } else {
                let Value::Tensor(result) = result else {
                    panic!("expected rank-one fitted result")
                };
                assert_eq!(result.shape(), shape.as_slice());
                assert!(result
                    .materialize_f64()
                    .iter()
                    .all(|value| value.is_finite()));
            }
        }
    }
}
