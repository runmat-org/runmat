mod capabilities;
mod descriptor;
mod documentation;
mod entry;
#[cfg(test)]
mod tests;

pub use capabilities::*;
pub use descriptor::CDF_DESCRIPTOR;
pub use entry::CDF_CATALOG_ENTRY;

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::fitted_distribution_facts::infer(
        request,
        entry,
        super::fitted_distribution_facts::EvaluationForm::Cumulative,
    )
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CDF_CATALOG_ENTRY];
