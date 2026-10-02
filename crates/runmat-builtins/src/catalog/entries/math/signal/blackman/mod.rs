mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::BLACKMAN_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&BLACKMAN_CATALOG_ENTRY];
