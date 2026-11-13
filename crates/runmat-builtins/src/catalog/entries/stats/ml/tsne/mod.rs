mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::TSNE_CATALOG_ENTRY;
pub use options::{
    TsneAlgorithm, TsneCacheSizeKeyword, TsneDisplay, TsneDistance, TsneOption, TsneStatsetField,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&TSNE_CATALOG_ENTRY];
