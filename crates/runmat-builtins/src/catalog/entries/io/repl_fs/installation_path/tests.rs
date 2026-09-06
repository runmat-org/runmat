use crate::catalog::infer_catalog_call;
use crate::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, ValueFact,
    ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn matlabroot_is_a_complete_identity_local_authority() {
    assert_eq!(MATLABROOT_CATALOG_ENTRY.identity.name, "matlabroot");
    assert_eq!(
        MATLABROOT_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Complete
    );
    assert_eq!(
        MATLABROOT_CATALOG_ENTRY.contract.inference_rule,
        BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(IoReplFsInferenceRule::Path(
            PathInferenceRule::Installation(InstallationPathInferenceRule::Root)
        )))
    );
    assert_eq!(
        MATLABROOT_CATALOG_ENTRY.documentation.authority,
        BuiltinDocumentationAuthority::Catalog
    );
    assert!(!MATLABROOT_CATALOG_ENTRY.documentation.examples.is_empty());
}

#[test]
fn inference_returns_a_character_row_and_rejects_inputs() {
    let valid = infer_catalog_call(&MATLABROOT_CATALOG_ENTRY, &request(Vec::new()));
    assert_eq!(valid.outputs.len(), 1);
    assert_eq!(valid.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(valid.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
    assert!(valid.diagnostics.is_empty());

    let invalid = infer_catalog_call(
        &MATLABROOT_CATALOG_ENTRY,
        &request(vec![ValueFact::scalar(ValueKindFact::Logical)]),
    );
    assert_eq!(invalid.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(invalid.diagnostics.len(), 1);
}
