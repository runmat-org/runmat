//! Dataset receiver identities.

mod accessors;
mod attributes;
mod lifecycle;
#[cfg(test)]
mod tests;

pub use accessors::*;
pub use attributes::*;
pub use lifecycle::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DATASET_ARRAY_CATALOG_ENTRY,
    &DATASET_ARRAYS_CATALOG_ENTRY,
    &DATASET_ATTRS_CATALOG_ENTRY,
    &DATASET_BEGIN_CATALOG_ENTRY,
    &DATASET_GET_ATTR_CATALOG_ENTRY,
    &DATASET_HAS_ARRAY_CATALOG_ENTRY,
    &DATASET_ID_CATALOG_ENTRY,
    &DATASET_PATH_CATALOG_ENTRY,
    &DATASET_REFRESH_CATALOG_ENTRY,
    &DATASET_SET_ATTR_CATALOG_ENTRY,
    &DATASET_SET_ATTRS_CATALOG_ENTRY,
    &DATASET_SNAPSHOT_CATALOG_ENTRY,
    &DATASET_VERSION_CATALOG_ENTRY,
];
