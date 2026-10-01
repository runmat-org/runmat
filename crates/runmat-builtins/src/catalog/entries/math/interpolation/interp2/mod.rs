mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::{
    INTERP2_DESCRIPTOR, INTERP2_ERROR_INTERNAL, INTERP2_ERROR_INVALID_ARGUMENT,
    INTERP2_ERROR_INVALID_INPUT,
};
pub use entry::INTERP2_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&INTERP2_CATALOG_ENTRY];
