mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::{
    POLYDER_DESCRIPTOR, POLYDER_ERROR_INTERNAL, POLYDER_ERROR_INVALID_ARGUMENT,
    POLYDER_ERROR_INVALID_INPUT, POLYDER_EXTENSIONS, POLYDER_INTEGER_CAPABILITIES,
    POLYDER_INTEGER_COEFFICIENTS_EXTENSION,
};
pub use entry::POLYDER_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&POLYDER_CATALOG_ENTRY];
