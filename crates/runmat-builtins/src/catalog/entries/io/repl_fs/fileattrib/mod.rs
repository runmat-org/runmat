mod contract;
mod control;
mod documentation;
mod entry;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FILEATTRIB_CATALOG_ENTRY];

pub use contract::{
    FILEATTRIB_DESCRIPTOR, FILEATTRIB_ERROR_TOO_MANY_OUTPUTS,
    FILEATTRIB_ERROR_UNSUPPORTED_PERMISSION, FILEATTRIB_INTEGER_AUDIT,
};
pub use control::{FileAttributeControl, FileAttributeControlError};
pub use entry::FILEATTRIB_CATALOG_ENTRY;
