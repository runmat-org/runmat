mod capabilities;
mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use super::fitted_distribution_model::{FittedDistributionFamily, FittedFitOption};
pub use capabilities::*;
pub use descriptor::FITDIST_DESCRIPTOR;
pub use entry::FITDIST_CATALOG_ENTRY;

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::infer(request, entry)
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FITDIST_CATALOG_ENTRY];
