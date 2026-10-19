mod inference;
mod pskmod;
mod qammod;

pub use pskmod::*;
pub use qammod::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&PSKMOD_CATALOG_ENTRY, &QAMMOD_CATALOG_ENTRY];
