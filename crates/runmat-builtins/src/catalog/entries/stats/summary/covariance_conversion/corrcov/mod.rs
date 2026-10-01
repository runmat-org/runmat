mod entry;
mod documentation;
#[cfg(test)]
mod tests;

pub use entry::CORRCOV_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CORRCOV_CATALOG_ENTRY];

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(request, entry, super::inference::ConversionKind::Corrcov)
}
