use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

#[test]
fn inspection_has_typed_properties_and_rejects_nontext_path() {
    let request = |fact| CallRequest {
        arguments: vec![fact],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let valid = inference::infer(
        &request(ValueFact::scalar(ValueKindFact::String)),
        &GEOMETRY_INSPECT_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    let ValueKindFact::Object(object) = &valid.outputs[0].kind else {
        panic!("expected inspect object")
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&GEOMETRY_INSPECT_RESULT_CLASS.owned())
    );
    assert!(object.properties.contains_key("format"));
    assert!(object.properties.contains_key("byte_count"));
    let bad = inference::infer(
        &request(ValueFact::scalar(ValueKindFact::Logical)),
        &GEOMETRY_INSPECT_CATALOG_ENTRY,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-GEOMETRY-INSPECT-PATH"));
}

#[test]
fn inspect_rejects_known_multirow_and_multistring_paths() {
    for fact in [
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ),
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
    ] {
        let result = inference::infer(
            &CallRequest {
                arguments: vec![fact],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &GEOMETRY_INSPECT_CATALOG_ENTRY,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-GEOMETRY-INSPECT-PATH"));
    }
}
