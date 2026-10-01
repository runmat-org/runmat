mod contract;
mod inference;
mod options;
mod pdist;
mod pdist2;

pub use contract::*;
pub use options::{DistanceControlKind, DistanceMetricKind, DistanceSelectionKind};
pub use pdist::PDIST_CATALOG_ENTRY;
pub use pdist2::PDIST2_CATALOG_ENTRY;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(pdist::ENTRIES.iter().copied());
    values.extend(pdist2::ENTRIES.iter().copied());
}
