use super::*;
use crate::*;

#[test]
fn documented_remaining_constructors_have_valid_unique_canonical_metadata() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(entries.len(), 14);
    assert_eq!(validate_builtin_catalog(&entries), Vec::new());
    for entry in entries {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(entry.bindings.len(), 1);
        assert_eq!(entry.descriptor.signatures.len(), 1);
        assert_eq!(entry.documentation.examples.len(), 1);
        assert!(entry.documentation.example_exemption.is_none());
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Incomplete);
        assert_eq!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::MaySuspend
        );
        assert_eq!(entry.contract.purity, BuiltinPurity::Impure);
        assert!(entry
            .contract
            .effects
            .contains(&runmat_types::EffectKind::Unknown));
    }
}
