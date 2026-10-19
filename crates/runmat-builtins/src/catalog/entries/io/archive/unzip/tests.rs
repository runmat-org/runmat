use super::*;
use runmat_types::*;

fn request(source: ValueFact, literal: Option<&str>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![source],
        literals: LiteralContext::new(vec![literal.map_or(LiteralValue::Unknown, |text| {
            LiteralValue::String(text.into())
        })]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn source_route_is_shared_with_runtime_and_refines_ambient_services() {
    assert!(matches!(
        UnzipSourceRoute::parse("HTTPS://example.test/a.zip".into()),
        UnzipSourceRoute::Http(_)
    ));
    assert!(matches!(
        UnzipSourceRoute::parse("C:\\data\\a.zip".into()),
        UnzipSourceRoute::LocalPath(_)
    ));
    assert!(matches!(
        UnzipSourceRoute::parse("ftp:relative.zip".into()),
        UnzipSourceRoute::LocalPath(_)
    ));
    let source = ValueFact::scalar(ValueKindFact::String);
    let local = inference::infer(
        &request(source.clone(), Some("a.zip"), 1),
        &UNZIP_CATALOG_ENTRY,
    );
    assert!(local
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(!local
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
    let http = inference::infer(
        &request(source.clone(), Some("https://example.test/a.zip"), 1),
        &UNZIP_CATALOG_ENTRY,
    );
    assert!(http
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
    assert!(http
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    let dynamic = inference::infer(&request(source, None, 1), &UNZIP_CATALOG_ENTRY);
    assert!(dynamic
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(dynamic
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
}

#[test]
fn result_shape_and_output_controls_match_runtime() {
    let source = ValueFact::unknown(DynamicReason::RuntimeValue);
    let one = inference::infer(&request(source.clone(), None, 1), &UNZIP_CATALOG_ENTRY);
    assert!(one.diagnostics.is_empty(), "{:?}", one.diagnostics);
    assert_eq!(one.outputs[0].shape, ShapeFact::from(vec![None, Some(1)]));
    assert_eq!(one.outputs[0].storage, StorageFact::Dense);
    assert_eq!(one.outputs[0].residency, ResidencyFact::Host);
    assert!(matches!(one.outputs[0].kind, ValueKindFact::Cell(_)));
    let zero = inference::infer(&request(source.clone(), None, 0), &UNZIP_CATALOG_ENTRY);
    assert!(zero.outputs.is_empty());
    assert!(zero.diagnostics.is_empty(), "{:?}", zero.diagnostics);
    let excess = inference::infer(&request(source, None, 2), &UNZIP_CATALOG_ENTRY);
    assert!(!excess.diagnostics.is_empty());
}

#[test]
fn invalid_text_and_empty_literal_are_diagnosed_without_rejecting_possible_char_rows() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(
        !inference::infer(&request(numeric, None, 1), &UNZIP_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let paged = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(7), Some(2)]),
        StorageFact::Dense,
    );
    assert!(
        !inference::infer(&request(paged, None, 1), &UNZIP_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    for shape in [
        ShapeFact::from(vec![Some(1), Some(7), Some(1)]),
        ShapeFact::from(vec![Some(1), None, None]),
        ShapeFact::Ranked { rank: 3 },
        ShapeFact::Unknown,
    ] {
        let possible = ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense);
        let inferred = inference::infer(&request(possible, None, 1), &UNZIP_CATALOG_ENTRY);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
    }
    let empty = inference::infer(
        &request(ValueFact::scalar(ValueKindFact::String), Some(" "), 1),
        &UNZIP_CATALOG_ENTRY,
    );
    assert!(!empty.diagnostics.is_empty());

    let mut destination = request(
        ValueFact::scalar(ValueKindFact::String),
        Some("sample.zip"),
        1,
    );
    destination
        .arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    destination
        .literals
        .literal_args
        .push(LiteralValue::String("out".into()));
    let valid = inference::infer(&destination, &UNZIP_CATALOG_ENTRY);
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    destination.arguments[1] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!inference::infer(&destination, &UNZIP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
