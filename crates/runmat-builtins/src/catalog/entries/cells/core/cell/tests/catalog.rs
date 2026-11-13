use super::*;
use crate::BuiltinInferenceRule;

#[test]
fn entry_owns_inference_signatures_and_executable_documentation() {
    assert!(matches!(
        CELL_CATALOG_ENTRY.contract.inference_rule,
        BuiltinInferenceRule::Identity(_)
    ));
    assert_eq!(CELL_CATALOG_ENTRY.documentation.examples.len(), 7);
    assert_eq!(CELL_CATALOG_ENTRY.descriptor.signatures.len(), 7);
    assert_eq!(
        CELL_CATALOG_ENTRY.descriptor.signatures[5].label,
        "C = cell(sz, \"like\", prototype)"
    );
    assert_eq!(
        CELL_CATALOG_ENTRY.descriptor.signatures[6].label,
        "C = cell(m, n, ..., \"like\", prototype)"
    );
    assert!(CELL_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(
            example.verification,
            crate::BuiltinExampleVerification::Succeeds
        )));
}
