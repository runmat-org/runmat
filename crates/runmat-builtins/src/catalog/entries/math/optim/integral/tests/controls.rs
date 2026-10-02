use super::super::*;
use super::support::*;
use runmat_types::*;

#[test]
fn interval_count_is_a_positive_structural_control_not_an_evaluation_alias() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let name = ValueFact::scalar(ValueKindFact::String);
    let call = |count: i64| {
        request(
            vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                real.clone(),
                real.clone(),
                name.clone(),
                real.clone(),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("MaxIntervalCount".into()),
                LiteralValue::Number(count as f64),
            ],
            1,
        )
    };
    let invalid = inference::infer(&call(0), &INTEGRAL_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-OPTION-VALUE"));
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
    let valid = inference::infer(&call(1), &INTEGRAL_CATALOG_ENTRY);
    assert!(valid.diagnostics.is_empty());
    assert!(matches!(valid.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn option_identity_is_typed_at_the_public_boundary() {
    for (spelling, identity) in [
        ("AbsTol", IntegralOptionName::AbsTol),
        ("reltol", IntegralOptionName::RelTol),
        ("AbsoluteTolerance", IntegralOptionName::AbsoluteTolerance),
        ("relativetolerance", IntegralOptionName::RelativeTolerance),
        ("MaxFunEvals", IntegralOptionName::MaxFunEvals),
        ("MaxIntervalCount", IntegralOptionName::MaxIntervalCount),
        ("ArrayValued", IntegralOptionName::ArrayValued),
        ("Vectorized", IntegralOptionName::Vectorized),
        ("Waypoints", IntegralOptionName::Waypoints),
    ] {
        assert_eq!(IntegralOptionName::parse(spelling), Some(identity));
    }
    assert_eq!(IntegralOptionName::parse("unknown"), None);
}

#[test]
fn known_invalid_controls_do_not_prove_success_and_last_override_is_respected() {
    let callback = ValueFact::unknown(DynamicReason::RuntimeValue);
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let text = ValueFact::scalar(ValueKindFact::String);
    for (name, value, fact) in [
        ("MaxFunEvals", LiteralValue::Number(4.0), real.clone()),
        ("ArrayValued", LiteralValue::Number(2.0), real.clone()),
    ] {
        let call = request(
            vec![
                callback.clone(),
                real.clone(),
                real.clone(),
                text.clone(),
                fact,
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String(name.into()),
                value,
            ],
            1,
        );
        let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-OPTION-VALUE"));
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    }

    let mut call = request(
        vec![
            callback,
            real.clone(),
            real.clone(),
            text.clone(),
            real.clone(),
            text.clone(),
            real.clone(),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("AbsTol".into()),
            LiteralValue::Number(0.0),
            LiteralValue::String("RelTol".into()),
            LiteralValue::Number(0.0),
        ],
        1,
    );
    let invalid = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-TOLERANCES"));
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));

    call.arguments.extend([text.clone(), real]);
    call.literals.literal_args.extend([
        LiteralValue::String("RelativeTolerance".into()),
        LiteralValue::Number(1.0e-6),
    ]);
    let overridden = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(overridden.diagnostics.is_empty());
    assert!(matches!(overridden.outputs[0].kind, ValueKindFact::Unknown));
}
