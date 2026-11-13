mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::FZERO_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FZERO_CATALOG_ENTRY];
