mod descriptor;
mod extensions;
mod integer;
mod entry;
mod inference;
mod documentation;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::CONFUSIONMAT_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CONFUSIONMAT_CATALOG_ENTRY];
