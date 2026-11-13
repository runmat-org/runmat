use super::*;
use runmat_types::*;

fn server() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::super::TCPSERVER_CLASS.owned()),
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: Some(true),
    }))
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
fn accepted_client_port_is_uint16_and_option_accepts_character_row() {
    let mut option = ValueFact::scalar(ValueKindFact::Character);
    option.shape = ShapeFact::from(vec![Some(1), Some(7)]);
    let request = |name: ValueFact| CallRequest {
        arguments: vec![server(), name, double()],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("Timeout".into()),
            LiteralValue::Number(1.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&request(option.clone()), &ACCEPT_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("accepted client has sealed object facts");
    };
    assert_eq!(
        object.runtime_class,
        Some(super::super::TCPCLIENT_CLASS.owned())
    );
    assert_eq!(
        object
            .properties
            .get("Port")
            .and_then(ValueFact::numeric)
            .map(|n| n.class),
        Some(NumericClass::UInt16)
    );
    option.shape = ShapeFact::from(vec![Some(2), Some(7)]);
    let invalid = inference::infer(&request(option), &ACCEPT_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NET-ACCEPT-OPTION"));
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
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Object(_)));
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
