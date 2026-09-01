mod contracts;
mod documentation;

pub use contracts::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CONJ_CATALOG_ENTRY,
    &REAL_CATALOG_ENTRY,
    &IMAG_CATALOG_ENTRY,
];
