use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn entries_are_complete_identity_local_catalog_authorities() {
    for entry in ENTRIES {
        assert_eq!(
            entry.contract.maturity,
            crate::BuiltinContractMaturity::Complete
        );
        assert_eq!(
            entry.documentation.authority,
            crate::BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
        assert!(entry.integer_audit.is_some());
    }
}

#[test]
fn static_results_distinguish_metadata_from_character_names() {
    let dir = crate::infer_catalog_call(
        &DIR_CATALOG_ENTRY,
        &request(vec![ValueFact::scalar(ValueKindFact::String)]),
    );
    let ls = crate::infer_catalog_call(
        &LS_CATALOG_ENTRY,
        &request(vec![ValueFact::scalar(ValueKindFact::Character)]),
    );
    assert!(dir.diagnostics.is_empty());
    assert!(matches!(dir.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(ls.outputs[0].kind, ValueKindFact::Character);
}

#[test]
fn known_invalid_types_and_arity_are_diagnostics() {
    let invalid = crate::infer_catalog_call(
        &LS_CATALOG_ENTRY,
        &request(vec![ValueFact::scalar(ValueKindFact::Logical)]),
    );
    let excess = crate::infer_catalog_call(
        &DIR_CATALOG_ENTRY,
        &request(vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ]),
    );
    assert!(!invalid.diagnostics.is_empty());
    assert!(!excess.diagnostics.is_empty());
}
