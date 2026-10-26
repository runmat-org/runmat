mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub use extensions::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LSQLIN_CATALOG_ENTRY];
