mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::KMEANS_CATALOG_ENTRY;
pub use options::{
    KmeansDisplay, KmeansDistance, KmeansEmptyAction, KmeansOption, KmeansStart,
    KmeansStatsetField, KmeansToggle,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&KMEANS_CATALOG_ENTRY];
