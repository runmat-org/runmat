mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::{
    POLYINT_DESCRIPTOR, POLYINT_ERROR_INVALID_ARGUMENT, POLYINT_ERROR_INVALID_INPUT,
    POLYINT_INTEGER_CAPABILITIES,
};
pub use entry::POLYINT_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&POLYINT_CATALOG_ENTRY];
