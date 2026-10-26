use super::*;
use crate::*;

#[test]
fn documented_remaining_constructors_have_valid_unique_canonical_metadata() {
    let mut entries = Vec::new();
    entries.extend(
        quantities::ENTRIES
            .iter()
            .chain(domains::ENTRIES)
            .chain(interfaces::ENTRIES)
            .copied(),
    );
    assert_eq!(entries.len(), 14);
    assert_eq!(validate_builtin_catalog(&entries), Vec::new());
    for entry in entries {
        let mut published = builtin_catalog_entries()
            .iter()
            .copied()
            .filter(|candidate| candidate.identity == entry.identity);
        let actual = published
            .next()
            .expect("canonical constructor is registered");
        assert!(
            published.next().is_none(),
            "constructor has one static owner"
        );
        assert_eq!(actual.provenance, entry.provenance);
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
