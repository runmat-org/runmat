mod descriptor;
mod extensions;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub use extensions::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LINPROG_CATALOG_ENTRY];
