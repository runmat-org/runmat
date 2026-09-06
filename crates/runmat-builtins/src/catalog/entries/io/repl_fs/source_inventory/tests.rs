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
fn what_is_a_complete_identity_local_authority() {
    assert_eq!(
        WHAT_CATALOG_ENTRY.contract.maturity,
        crate::BuiltinContractMaturity::Complete
    );
    assert_eq!(
        WHAT_CATALOG_ENTRY.documentation.authority,
        crate::BuiltinDocumentationAuthority::Catalog
    );
    assert!(!WHAT_CATALOG_ENTRY.documentation.examples.is_empty());
}

#[test]
fn inference_reports_the_complete_result_structure() {
    let inference = crate::infer_catalog_call(&WHAT_CATALOG_ENTRY, &request(Vec::new()));
    assert!(inference.diagnostics.is_empty());
    let ValueKindFact::Struct(result) = &inference.outputs[0].kind else {
        panic!("expected structure fact")
    };
    assert!(result.fields_complete);
    assert_eq!(
        result.fields.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["classes", "m", "mat", "mex", "packages", "path"]
    );
}

#[test]
fn inference_rejects_known_invalid_folder_and_arity() {
    let invalid = crate::infer_catalog_call(
        &WHAT_CATALOG_ENTRY,
        &request(vec![ValueFact::scalar(ValueKindFact::Logical)]),
    );
    let excess = crate::infer_catalog_call(
        &WHAT_CATALOG_ENTRY,
        &request(vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ]),
    );
    assert!(!invalid.diagnostics.is_empty());
    assert!(!excess.diagnostics.is_empty());
}
