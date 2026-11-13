//! Linear classifier fitting contract.
mod contract;
mod documentation;
mod entry;
mod inference;
mod integer;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::FITCLINEAR_CATALOG_ENTRY;
pub use integer::FITCLINEAR_INTEGER_CAPABILITIES;
pub use options::{FitclinearOption, FitclinearSolver};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FITCLINEAR_CATALOG_ENTRY];
