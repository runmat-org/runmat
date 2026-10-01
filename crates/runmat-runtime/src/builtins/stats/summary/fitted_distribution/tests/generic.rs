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
    assert!(DistributionSelection::from_value(FittedOperation::Density, &wrong_class).is_err());
    let normal = DistributionSelection::from_value(
        FittedOperation::Cumulative,
        &Value::String("Normal".into()),
    )
    .unwrap();
    assert!(admission::evaluation(
        CDF_NAME,
        &normal,
        &Value::Num(0.0),
        &[Value::Num(0.0), Value::Num(1.0), Value::Num(2.0)],
    )
    .is_err());
    assert!(admission::fit(
        &Value::Num(1.0),
        &Value::String("Normal".into()),
        &[Value::String("Unsupported".into()), Value::Num(1.0)],
    )
    .is_err());
    let gamma = DistributionSelection::from_value(
        FittedOperation::Sampling,
        &Value::String("Gamma".into()),
    )
    .unwrap();
    assert!(admission::random(&gamma, &[Value::Num(2.0)]).is_err());
}

#[test]
fn malformed_fitted_object_errors_use_the_invoked_public_operation() {
    let malformed = Value::Object(ObjectInstance::new(
        runmat_types::standard::PROBABILITY_DISTRIBUTION.to_string(),
    ));
    let pdf = DistributionSelection::from_value(FittedOperation::Density, &malformed)
        .err()
        .unwrap();
    assert_eq!(pdf.identifier(), Some("RunMat:pdf:InvalidArgument"));
    let cdf = DistributionSelection::from_value(FittedOperation::Cumulative, &malformed)
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
    let admitted = admission::fit(&Value::Num(2.0), &Value::from("norm"), &options)
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
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let density = block_on(pdf_builtin(
        Value::String("Normal".into()),
        mirrorless_int_tensor(IntegerStorage::I16(vec![0, 1]), vec![2, 1]),
        vec![
            mirrorless_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
            mirrorless_int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
        ],
    ))
    .unwrap();
    let Value::Tensor(density) = density else {
        panic!("expected vector density");
    };
    assert_eq!(density.shape, vec![2, 1]);
    assert!(
        (density.materialize_f64()[0] - distribution_math::standard_normal_pdf(0.0)).abs()
            < 1.0e-12
    );

    let probability = block_on(cdf_builtin(
        Value::String("Poisson".into()),
        Value::Num(2.0),
        vec![Value::Num(3.0)],
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
        Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 3]),
        other => panic!("expected tensor, got {other:?}"),
    }
}
