use super::*;
use runmat_types::*;

#[test]
fn address_and_option_admit_character_rows_not_matrices() {
    let mut address = ValueFact::scalar(ValueKindFact::Character);
    address.shape = ShapeFact::from(vec![Some(1), Some(9)]);
    let mut option = ValueFact::scalar(ValueKindFact::Character);
    option.shape = ShapeFact::from(vec![Some(1), Some(7)]);
    let request = |address: ValueFact, option: ValueFact| CallRequest {
        arguments: vec![
            address,
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            option,
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("localhost".into()),
            LiteralValue::Number(80.0),
            LiteralValue::String("Timeout".into()),
            LiteralValue::Number(5.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert!(inference::infer(
        &request(address.clone(), option.clone()),
        &TCPSERVER_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    address.shape = ShapeFact::from(vec![Some(2), Some(9)]);
    option.shape = ShapeFact::from(vec![Some(2), Some(7)]);
    let invalid = inference::infer(&request(address, option), &TCPSERVER_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPSERVER-ADDRESS"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TCPSERVER-OPTION"));
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
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("tcpserver returns a sealed host object");
    };
    assert_eq!(
        object.runtime_class,
        Some(super::super::TCPSERVER_CLASS.owned())
    );
    assert_eq!(
        object
            .properties
            .get("ServerPort")
            .and_then(ValueFact::numeric)
            .map(|n| n.class),
        Some(NumericClass::Double)
    );
    assert!(matches!(
        object.properties.get("Connected").map(|v| &v.kind),
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
