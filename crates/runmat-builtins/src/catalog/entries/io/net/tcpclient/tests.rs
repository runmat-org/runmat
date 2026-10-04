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
fn constructor_facts_preserve_host_property_classes_and_known_invalid_controls() {
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
    let inferred = inference::infer(&valid, &TCPCLIENT_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Struct(fields) = &inferred.outputs[0].kind else {
        panic!("tcpclient returns a host struct in the current runtime contract");
    };
    assert_eq!(
        fields
            .fields
            .get("Port")
            .and_then(ValueFact::numeric)
            .map(|n| n.class),
        Some(NumericClass::Double)
    );
    assert!(matches!(
        fields.fields.get("Connected").map(|v| &v.kind),
        Some(ValueKindFact::Logical)
    ));
    assert_eq!(TCPCLIENT_DESCRIPTOR.signatures.len(), 2);

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
    let inferred = inference::infer(&invalid, &TCPCLIENT_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPCLIENT-PORT"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPCLIENT-OPTION"));
}
