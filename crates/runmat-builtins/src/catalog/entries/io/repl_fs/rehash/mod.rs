mod contract;
mod documentation;
mod entry;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&REHASH_CATALOG_ENTRY];

pub use contract::{
    RehashMode, REHASH_DESCRIPTOR, REHASH_ERROR_INVALID_ARITY, REHASH_ERROR_INVALID_MODE,
    REHASH_ERROR_TOO_MANY_OUTPUTS, REHASH_ERROR_UNAVAILABLE, REHASH_INTEGER_AUDIT,
};
pub use entry::REHASH_CATALOG_ENTRY;
