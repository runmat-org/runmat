mod nextpow2;
mod pow2;

pub use nextpow2::*;
pub use pow2::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&NEXTPOW2_CATALOG_ENTRY, &POW2_CATALOG_ENTRY];
