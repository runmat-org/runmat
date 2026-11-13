mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::HILBERT_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&HILBERT_CATALOG_ENTRY];
