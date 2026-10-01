mod capabilities;
mod documentation;
mod entry;
#[cfg(test)]
mod tests;

pub use capabilities::*;
pub use entry::TCDF_CATALOG_ENTRY;

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(request, entry, super::policy::DistributionKind::StudentTCdf)
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&TCDF_CATALOG_ENTRY];
