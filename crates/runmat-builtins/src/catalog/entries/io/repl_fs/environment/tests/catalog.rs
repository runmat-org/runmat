use crate::{
    BuiltinContractMaturity, BuiltinDocumentationAuthority, BuiltinExampleVerification,
    GETENV_CATALOG_ENTRY, ISENV_CATALOG_ENTRY, SETENV_CATALOG_ENTRY, UNSETENV_CATALOG_ENTRY,
};

#[test]
fn every_environment_identity_has_complete_catalog_owned_documentation() {
    for entry in [
        &GETENV_CATALOG_ENTRY,
        &SETENV_CATALOG_ENTRY,
        &ISENV_CATALOG_ENTRY,
        &UNSETENV_CATALOG_ENTRY,
    ] {
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Complete);
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.summary.is_empty());
        assert!(!entry.documentation.examples.is_empty());
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
    }
}
