mod descriptor;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::FMINSEARCH_CATALOG_ENTRY;
pub use options::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FMINSEARCH_CATALOG_ENTRY];
