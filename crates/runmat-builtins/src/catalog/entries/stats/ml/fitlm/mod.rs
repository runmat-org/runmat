//! Linear-model fitting contract and public option identities.
mod contract;
mod documentation;
mod entry;
mod inference;
mod integer;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::FITLM_CATALOG_ENTRY;
pub use integer::FITLM_INTEGER_CAPABILITIES;
pub use options::FitlmOption;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FITLM_CATALOG_ENTRY];
