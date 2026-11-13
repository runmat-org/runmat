mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::{
    POLY_DESCRIPTOR, POLY_ERROR_INVALID_INPUT, POLY_ERROR_NUMERICAL,
    POLY_ERROR_UNSUPPORTED_PROVIDER, POLY_INTEGER_CAPABILITIES,
};
pub use entry::POLY_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&POLY_CATALOG_ENTRY];
