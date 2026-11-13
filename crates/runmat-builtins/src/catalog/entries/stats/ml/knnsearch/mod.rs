mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::KNNSEARCH_CATALOG_ENTRY;
pub use options::{KnnCacheSizeKeyword, KnnMetric, KnnOption, KnnSearchMethod};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&KNNSEARCH_CATALOG_ENTRY];
