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
        PDF_NAME,
        &wrong_class,
        &Value::Num(0.0),
        &[],
        admission::EvaluationKind::Density,
    )
    .is_err());
    assert!(admission::evaluation(
        CDF_NAME,
        &Value::String("Normal".into()),
        &Value::Num(0.0),
        &[Value::Num(0.0), Value::Num(1.0), Value::Num(2.0)],
        admission::EvaluationKind::Cumulative,
    )
    .is_err());
    assert!(admission::fit(
        &Value::Num(1.0),
        &Value::String("Normal".into()),
        &[Value::String("Unsupported".into()), Value::Num(1.0)],
    )
    .is_err());
    assert!(admission::random(&Value::String("Gamma".into()), &[Value::Num(2.0)],).is_err());
}

#[test]
fn malformed_fitted_object_errors_use_the_invoked_public_operation() {
    let malformed = Value::Object(ObjectInstance::new(
        runmat_types::standard::PROBABILITY_DISTRIBUTION.to_string(),
    ));
    let pdf = admission::evaluation(
        PDF_NAME,
        &malformed,
        &Value::Num(0.0),
        &[],
        admission::EvaluationKind::Density,
    )
    .unwrap_err();
    assert_eq!(pdf.identifier(), Some("RunMat:pdf:InvalidArgument"));
    let cdf = admission::evaluation(
        CDF_NAME,
        &malformed,
        &Value::Num(0.0),
        &[],
        admission::EvaluationKind::Cumulative,
    )
    .unwrap_err();
    assert_eq!(cdf.identifier(), Some("RunMat:cdf:InvalidArgument"));
    let random = admission::random(&malformed, &[]).unwrap_err();
    assert_eq!(random.identifier(), Some("RunMat:random:InvalidArgument"));
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
