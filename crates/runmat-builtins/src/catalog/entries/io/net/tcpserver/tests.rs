use super::*;
use runmat_types::*;

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
fn server_constructor_has_sound_fields_and_invalid_control_diagnostics() {
    let valid = CallRequest {
        arguments: vec![text(), double(), text(), double()],
        literals: LiteralContext::new(vec![
            LiteralValue::String("127.0.0.1".into()),
            LiteralValue::Number(50000.0),
            LiteralValue::String("Timeout".into()),
            LiteralValue::Number(5.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&valid, &TCPSERVER_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Struct(fields) = &inferred.outputs[0].kind else {
        panic!("tcpserver returns a host struct in the current runtime contract");
    };
    assert_eq!(
        fields
            .fields
            .get("ServerPort")
            .and_then(ValueFact::numeric)
            .map(|n| n.class),
        Some(NumericClass::Double)
    );
    assert!(matches!(
        fields.fields.get("Connected").map(|v| &v.kind),
        Some(ValueKindFact::Logical)
    ));
    assert_eq!(TCPSERVER_DESCRIPTOR.signatures.len(), 2);

    let invalid = CallRequest {
        arguments: vec![text(), double(), text(), double()],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(70_000.0),
            LiteralValue::String("EnableTransferDelay".into()),
            LiteralValue::Number(1.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&invalid, &TCPSERVER_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPSERVER-PORT"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPSERVER-OPTION"));

    let infinite = CallRequest {
        arguments: vec![text(), double(), text(), double()],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
            LiteralValue::String("Timeout".into()),
            LiteralValue::Number(f64::INFINITY),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&infinite, &TCPSERVER_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPSERVER-TIMEOUT"));
}
