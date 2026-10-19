use crate::{
    BuiltinContractMaturity, BuiltinDocumentationAuthority, BuiltinExampleVerification,
    TEMPDIR_CATALOG_ENTRY, TEMPNAME_CATALOG_ENTRY,
};

#[test]
fn temporary_path_entries_are_complete_and_catalog_owned() {
    for entry in [&TEMPDIR_CATALOG_ENTRY, &TEMPNAME_CATALOG_ENTRY] {
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Complete);
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
    }
}
