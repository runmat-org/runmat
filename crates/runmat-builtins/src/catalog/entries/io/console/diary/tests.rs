use super::*;
use runmat_types::{
    CallRequest, CapabilityRequirement, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

#[test]
fn known_off_removes_file_write_but_filename_requires_filesystem() {
    let request = |literal: &str| CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::new(vec![LiteralValue::String(literal.into())]),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    };
    let off = inference::infer(&request("off"), &DIARY_CATALOG_ENTRY);
    assert!(!off
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    let file = inference::infer(&request("session.log"), &DIARY_CATALOG_ENTRY);
    assert!(file
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(inference::infer(&request("  "), &DIARY_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DIARY-EMPTY"));
}

#[test]
fn character_pages_reject_but_trailing_singleton_is_possible() {
    let request = |shape| CallRequest {
        arguments: vec![ValueFact::proven(
            ValueKindFact::Character,
            runmat_types::ShapeFact::from(shape),
            runmat_types::StorageFact::Dense,
        )],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    };
    let invalid = inference::infer(
        &request(vec![Some(1), Some(2), Some(2)]),
        &DIARY_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DIARY-TEXT"));
    let valid = inference::infer(
        &request(vec![Some(1), Some(2), Some(1)]),
        &DIARY_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty());
}
