use super::*;
use runmat_types::{ShapeFact, ValueKindFact};

#[test]
fn all_identities_are_complete_catalog_authorities_with_examples() {
    for name in ["fileparts", "filesep", "fullfile", "pathsep"] {
        let entry = crate::builtin_catalog_entry_by_name(name).expect("catalog entry");
        assert_eq!(
            entry.contract.maturity,
            crate::BuiltinContractMaturity::Complete
        );
        assert_eq!(
            entry.documentation.authority,
            crate::BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
    }
}

#[test]
fn separators_are_character_scalars_and_reject_inputs() {
    for name in ["filesep", "pathsep"] {
        let entry = crate::builtin_catalog_entry_by_name(name).expect("catalog entry");
        let result =
            crate::infer_catalog_call(entry, &request(Vec::new(), RequestedOutputCount::One));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(1)])
        );
        let invalid = crate::infer_catalog_call(
            entry,
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                RequestedOutputCount::One,
            ),
        );
        assert!(!invalid.diagnostics.is_empty());
    }
}
