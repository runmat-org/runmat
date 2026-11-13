mod contract;
mod documentation;
mod entry;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&RESTOREDEFAULTPATH_CATALOG_ENTRY];

pub use contract::{
    RESTOREDEFAULTPATH_DESCRIPTOR, RESTOREDEFAULTPATH_INTEGER_AUDIT,
    RESTOREDEFAULTPATH_RETURN_EXTENSION,
};
pub use entry::RESTOREDEFAULTPATH_CATALOG_ENTRY;
