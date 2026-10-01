mod contract;
mod inference;
mod options;
mod pdist;
mod pdist2;

pub use contract::*;
pub use options::{DistanceControlKind, DistanceMetricKind, DistanceSelectionKind};
pub use pdist::PDIST_CATALOG_ENTRY;
pub use pdist2::PDIST2_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&PDIST_CATALOG_ENTRY, &PDIST2_CATALOG_ENTRY];
