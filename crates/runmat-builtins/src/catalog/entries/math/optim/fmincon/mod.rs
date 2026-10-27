mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod reflective_documentation;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub use extensions::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FMINCON_CATALOG_ENTRY];
