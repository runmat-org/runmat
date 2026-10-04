use super::*;
use runmat_types::*;

fn server() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn accepted_client_fields_and_timeout_controls_are_sound() {
    let valid = CallRequest {
        arguments: vec![server(), text(), double()],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("Timeout".into()),
            LiteralValue::Number(0.25),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&valid, &ACCEPT_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Struct(_)));
    assert_eq!(ACCEPT_DESCRIPTOR.signatures.len(), 3);

    let invalid = CallRequest {
        arguments: vec![server(), text(), double()],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("Backlog".into()),
            LiteralValue::Number(-1.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&invalid, &ACCEPT_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NET-ACCEPT-OPTION"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NET-ACCEPT-TIMEOUT"));
}
