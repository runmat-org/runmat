mod descriptor;
mod documentation;
mod entry;
mod inference;
pub(super) use inference::inspect_descriptor;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SECONDORDERCONE_CATALOG_ENTRY];
