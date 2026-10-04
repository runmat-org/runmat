//! Dataset receiver identities.

mod accessors;
#[cfg(test)]
mod tests;

pub use accessors::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DATASET_ARRAY_CATALOG_ENTRY,
    &DATASET_ARRAYS_CATALOG_ENTRY,
    &DATASET_HAS_ARRAY_CATALOG_ENTRY,
    &DATASET_ID_CATALOG_ENTRY,
    &DATASET_PATH_CATALOG_ENTRY,
    &DATASET_VERSION_CATALOG_ENTRY,
];
