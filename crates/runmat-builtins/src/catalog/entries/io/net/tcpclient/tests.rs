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
fn host_and_option_names_admit_character_rows_but_reject_character_matrices() {
    let mut host = ValueFact::scalar(ValueKindFact::Character);
    host.shape = ShapeFact::from(vec![Some(1), Some(9)]);
    let mut option = ValueFact::scalar(ValueKindFact::Character);
    option.shape = ShapeFact::from(vec![Some(1), Some(7)]);
    let request = |host: ValueFact, option: ValueFact| CallRequest {
        arguments: vec![host, double(), option, double()],
        literals: LiteralContext::new(vec![
            LiteralValue::String("localhost".into()),
            LiteralValue::Number(80.0),
            LiteralValue::String("Timeout".into()),
            LiteralValue::Number(5.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert!(inference::infer(
        &request(host.clone(), option.clone()),
        &TCPCLIENT_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    host.shape = ShapeFact::from(vec![Some(2), Some(9)]);
    option.shape = ShapeFact::from(vec![Some(2), Some(7)]);
    let invalid = inference::infer(&request(host, option), &TCPCLIENT_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPCLIENT-HOST"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPCLIENT-OPTION"));
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
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("tcpclient returns a sealed host object");
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
        Some(NumericClass::Double)
    );
    assert!(matches!(
        object.properties.get("Connected").map(|v| &v.kind),
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

    let negative_timeout = CallRequest {
        arguments: vec![text(), double(), text(), double()],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(80.0),
            LiteralValue::String("Timeout".into()),
            LiteralValue::Number(-1.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&negative_timeout, &TCPCLIENT_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPCLIENT-TIMEOUT"));
}
