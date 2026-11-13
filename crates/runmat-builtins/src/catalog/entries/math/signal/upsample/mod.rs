mod descriptor;
mod documentation;
mod entry;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::UPSAMPLE_CATALOG_ENTRY;
pub use integer::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&UPSAMPLE_CATALOG_ENTRY];
