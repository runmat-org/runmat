mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&OPTIMGET_CATALOG_ENTRY];
