mod descriptor;
mod documentation;
mod entry;
mod inference;
mod integer;
mod options;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::SQUAREFORM_CATALOG_ENTRY;
pub use integer::*;
pub use options::{condensed_length, matrix_size_for_condensed_length, SquareformDirection};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SQUAREFORM_CATALOG_ENTRY];
