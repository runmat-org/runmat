mod descriptor;
mod documentation;
mod entry;
mod inference;
mod integer;
mod extensions;
pub use extensions::*;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::HISTCOUNTS2_CATALOG_ENTRY;
pub use integer::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&HISTCOUNTS2_CATALOG_ENTRY];
