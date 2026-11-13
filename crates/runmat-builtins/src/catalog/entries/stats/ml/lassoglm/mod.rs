//! Regularized generalized linear model public contract.
mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::LASSOGLM_CATALOG_ENTRY;
pub use options::{
    LassoglmDistribution, LassoglmEstimateDisp, LassoglmLink, LassoglmOption, LassoglmOptionRole,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LASSOGLM_CATALOG_ENTRY];
