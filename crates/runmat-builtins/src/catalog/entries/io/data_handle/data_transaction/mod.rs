//! Managed transaction identities and the standalone commit alias.

#[macro_use]
mod entry;

mod accessors;
mod completion;
mod queue;
#[cfg(test)]
mod tests;

pub use accessors::*;
pub use completion::*;
pub use queue::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DATATX_ABORT_CATALOG_ENTRY,
    &DATATX_COMMIT_CATALOG_ENTRY,
    &DATATX_CREATE_ARRAY_CATALOG_ENTRY,
    &DATATX_DELETE_ARRAY_CATALOG_ENTRY,
    &DATATX_FILL_CATALOG_ENTRY,
    &DATATX_ID_CATALOG_ENTRY,
    &DATATX_RESIZE_CATALOG_ENTRY,
    &DATATX_SET_ATTR_CATALOG_ENTRY,
    &DATATX_SET_ATTRS_CATALOG_ENTRY,
    &DATATX_STATUS_CATALOG_ENTRY,
    &DATATX_WRITE_CATALOG_ENTRY,
    &COMMIT_ALIAS_CATALOG_ENTRY,
];
