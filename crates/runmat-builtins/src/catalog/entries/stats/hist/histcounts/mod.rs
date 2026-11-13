mod descriptor;
mod documentation;
mod entry;
mod inference;
mod integer;
mod options;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::HISTCOUNTS_CATALOG_ENTRY;
pub use integer::*;
pub use options::HistcountsOption;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&HISTCOUNTS_CATALOG_ENTRY];
