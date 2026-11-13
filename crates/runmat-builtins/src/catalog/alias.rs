use super::{BuiltinCatalogIdentity, BuiltinCatalogProvenance};
use serde::Serialize;

/// One public spelling that resolves to a canonical builtin catalog identity.
///
/// Aliases own no contract, documentation, inference, placement, or runtime
/// binding. Every consumer resolves the spelling to `canonical` and uses that
/// entry as the sole semantic and executable authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinCatalogAlias {
    pub alias: BuiltinCatalogIdentity,
    pub canonical: BuiltinCatalogIdentity,
    pub provenance: BuiltinCatalogProvenance,
}

impl BuiltinCatalogAlias {
    pub const fn with_provenance(
        alias: &'static str,
        canonical: &'static str,
        provenance: BuiltinCatalogProvenance,
    ) -> Self {
        Self {
            alias: BuiltinCatalogIdentity { name: alias },
            canonical: BuiltinCatalogIdentity { name: canonical },
            provenance,
        }
    }
}
