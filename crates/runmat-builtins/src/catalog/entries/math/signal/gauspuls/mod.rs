mod descriptor;
mod extensions;
mod integer;
mod inference;
mod documentation;
mod entry;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::GAUSPULS_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&GAUSPULS_CATALOG_ENTRY];
