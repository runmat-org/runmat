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
pub use options::SquareformDirection;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SQUAREFORM_CATALOG_ENTRY];
