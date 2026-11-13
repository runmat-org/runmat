pub mod gamma;
pub mod gammaln;

pub use gamma::*;
pub use gammaln::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&gamma::GAMMA_CATALOG_ENTRY, &gammaln::GAMMALN_CATALOG_ENTRY];
