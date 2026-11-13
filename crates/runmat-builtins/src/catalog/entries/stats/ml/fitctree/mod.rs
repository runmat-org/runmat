//! Classification-tree fitting contract.
mod contract;
mod documentation;
mod entry;
mod inference;
mod integer;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::FITCTREE_CATALOG_ENTRY;
pub use integer::FITCTREE_INTEGER_CAPABILITIES;
pub use options::FitctreeOption;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FITCTREE_CATALOG_ENTRY];
