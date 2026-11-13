mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::{
    ROOTS_DESCRIPTOR, ROOTS_ERROR_INTERNAL, ROOTS_ERROR_INVALID_INPUT, ROOTS_EXTENSIONS,
    ROOTS_INTEGER_CAPABILITIES, ROOTS_INTEGER_COEFFICIENTS_EXTENSION,
};
pub use entry::ROOTS_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ROOTS_CATALOG_ENTRY];
