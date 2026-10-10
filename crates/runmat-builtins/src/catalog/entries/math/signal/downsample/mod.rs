mod descriptor;
mod documentation;
mod entry;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::DOWNSAMPLE_CATALOG_ENTRY;
pub use integer::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&DOWNSAMPLE_CATALOG_ENTRY];
