mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::ZPLANE_CATALOG_ENTRY;
pub use extensions::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ZPLANE_CATALOG_ENTRY];
