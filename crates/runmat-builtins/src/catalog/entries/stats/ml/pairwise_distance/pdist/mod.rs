mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use entry::PDIST_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PDIST_CATALOG_ENTRY];
