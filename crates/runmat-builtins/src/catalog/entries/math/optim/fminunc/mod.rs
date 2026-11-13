mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::FMINUNC_CATALOG_ENTRY;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FMINUNC_CATALOG_ENTRY];
