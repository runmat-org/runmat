mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::{
    OPTIMSET_DESCRIPTOR, OPTIMSET_ERROR_INVALID_ARGUMENT, OPTIMSET_ERROR_INVALID_INPUT,
    OPTIMSET_EXTENSIONS, OPTIMSET_INTEGER_CAPABILITIES, OPTIMSET_INTEGER_OPTION_EXTENSION,
    OPTIMSET_RESIDENT_OPTION_EXTENSION,
};
pub use entry::OPTIMSET_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&OPTIMSET_CATALOG_ENTRY];
