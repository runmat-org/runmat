mod capabilities;
mod documentation;
mod entry;
#[cfg(test)]
mod tests;

pub use capabilities::*;
pub use entry::ICDF_CATALOG_ENTRY;

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(request, entry, super::policy::DistributionKind::GenericInv)
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ICDF_CATALOG_ENTRY];
