mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::{
    INTERP1_DESCRIPTOR, INTERP1_ERROR_INTERNAL, INTERP1_ERROR_INVALID_ARGUMENT,
    INTERP1_ERROR_INVALID_INPUT,
};
pub use entry::INTERP1_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&INTERP1_CATALOG_ENTRY];
