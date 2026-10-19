mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::LINKAGE_CATALOG_ENTRY;
pub use options::{LinkageMethod, LinkageOption, SaveMemorySetting};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LINKAGE_CATALOG_ENTRY];
