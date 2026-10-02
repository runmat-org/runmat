use super::*;
use runmat_types::*;

fn scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn finite_scalar_bounds_prove_host_double_result() {
    let call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(NumericClass::Double, NumericDomain::Real),
            scalar(NumericClass::Double, NumericDomain::Real),
        ],
        vec![],
        1,
    );
    let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn invalid_complex_bound_and_incomplete_option_pairs_do_not_prove_success() {
    let mut call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(NumericClass::Double, NumericDomain::Complex),
            scalar(NumericClass::Double, NumericDomain::Real),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("AbsTol".into()),
        ],
        1,
    );
    let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(!inferred.diagnostics.is_empty());
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    call.arguments[1] = scalar(NumericClass::Double, NumericDomain::Real);
    call.arguments
        .push(scalar(NumericClass::Double, NumericDomain::Real));
    call.literals.literal_args.push(LiteralValue::Unknown);
    let valid = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(valid.diagnostics.is_empty());
    assert_eq!(
        valid.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn callback_capability_is_transitive() {
    let mut call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(NumericClass::Double, NumericDomain::Real),
            scalar(NumericClass::Double, NumericDomain::Real),
        ],
        vec![],
        1,
    );
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet(std::collections::BTreeSet::from([
            CapabilityRequirement::Filesystem,
        ])),
        parameters: vec![scalar(NumericClass::Double, NumericDomain::Real)],
        parameters_complete: true,
        outputs: vec![scalar(NumericClass::Double, NumericDomain::Real)],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn option_identity_is_typed_at_the_public_boundary() {
    for (spelling, identity) in [
        ("AbsTol", IntegralOptionName::AbsTol),
        ("reltol", IntegralOptionName::RelTol),
        ("MaxFunEvals", IntegralOptionName::MaxFunEvals),
        ("MaxIntervalCount", IntegralOptionName::MaxIntervalCount),
        ("ArrayValued", IntegralOptionName::ArrayValued),
    ] {
        assert_eq!(IntegralOptionName::parse(spelling), Some(identity));
    }
    assert_eq!(IntegralOptionName::parse("unknown"), None);
}
