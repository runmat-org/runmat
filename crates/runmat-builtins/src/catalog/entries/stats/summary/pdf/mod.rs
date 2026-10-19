mod capabilities;
mod descriptor;
mod documentation;
mod entry;
#[cfg(test)]
mod tests;

pub use capabilities::*;
pub use descriptor::PDF_DESCRIPTOR;
pub use entry::PDF_CATALOG_ENTRY;

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::fitted_distribution_facts::infer(
        request,
        entry,
        super::fitted_distribution_facts::EvaluationForm::Density,
    )
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PDF_CATALOG_ENTRY];
