use super::*;
use runmat_types::*;

fn signal(class: NumericClass, domain: NumericDomain, residency: ResidencyFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(vec![Some(128), Some(2)]),
        StorageFact::Dense,
    );
    fact.residency = residency;
    fact
}

fn infer(input: ValueFact, outputs: RequestedOutputCount) -> CallInference {
    crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(outputs),
        },
    )
}

#[test]
fn parent_literal_requires_a_scalar_axes_candidate() {
    let signal = signal(
        NumericClass::Double,
        NumericDomain::Real,
        ResidencyFact::Host,
    );
    let name = ValueFact::scalar(ValueKindFact::String);
    let invalid_target = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let result = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![signal, name, invalid_target],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("Parent".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PWELCH-PARENT"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn output_class_is_signal_derived_and_residency_is_not_proven() {
    for class in [
        NumericClass::Single,
        NumericClass::Double,
        NumericClass::Int64,
    ] {
        let input = signal(class, NumericDomain::Real, ResidencyFact::Host);
        let facts = infer(input, RequestedOutputCount::Exactly(3));
        let expected = if class == NumericClass::Single {
            NumericClass::Single
        } else {
            NumericClass::Double
        };
        assert_eq!(facts.outputs[0].numeric().unwrap().class, expected);
        assert_eq!(facts.outputs[1].numeric().unwrap().class, expected);
        assert_eq!(
            facts.outputs[0].shape.known_dims(),
            Some(vec![None, Some(2)])
        );
        assert_eq!(
            facts.outputs[1].shape.known_dims(),
            Some(vec![None, Some(1)])
        );
        assert_eq!(
            facts.outputs[2].numeric().unwrap().class,
            NumericClass::Double
        );
        assert!(facts.outputs[2].is_scalar());
    }
    let input = signal(
        NumericClass::Single,
        NumericDomain::Complex,
        ResidencyFact::Device {
            provider: Some("owner".into()),
        },
    );
    let facts = infer(input, RequestedOutputCount::Exactly(2));
    assert!(facts
        .outputs
        .iter()
        .all(|fact| fact.residency == ResidencyFact::Unknown));
    assert!(facts
        .outputs
        .iter()
        .all(|fact| fact.numeric().unwrap().domain == NumericDomain::Real));
}

#[test]
fn invalid_signal_does_not_prove_outputs() {
    let input = signal(
        NumericClass::Int16,
        NumericDomain::Complex,
        ResidencyFact::Host,
    );
    let facts = infer(input, RequestedOutputCount::One);
    assert!(!facts.diagnostics.is_empty());
    assert_eq!(facts.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn known_invalid_control_does_not_prove_result() {
    let result = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                signal(
                    NumericClass::Single,
                    NumericDomain::Real,
                    ResidencyFact::Host,
                ),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(-1.0)]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn complex_onesided_selector_has_no_success_fact() {
    let input = signal(
        NumericClass::Double,
        NumericDomain::Complex,
        ResidencyFact::Host,
    );
    let result = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                input,
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(8.0),
                LiteralValue::Number(0.0),
                LiteralValue::Number(8.0),
                LiteralValue::String("onesided".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PWELCH-RANGE"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn descriptor_and_docs_are_canonical() {
    assert!(crate::validate_builtin_catalog(&[&PWELCH_CATALOG_ENTRY]).is_empty());
    assert_eq!(PWELCH_CATALOG_ENTRY.identity.name, "pwelch");
    assert_eq!(PWELCH_CATALOG_ENTRY.documentation.examples.len(), 8);
    assert_eq!(
        PWELCH_CATALOG_ENTRY.descriptor.signatures[1].outputs.len(),
        2
    );
}
