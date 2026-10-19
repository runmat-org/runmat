mod capabilities;
mod documentation;
mod entry;
#[cfg(test)]
mod tests;

pub use capabilities::*;
pub use entry::NORMINV_CATALOG_ENTRY;

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(request, entry, super::policy::DistributionKind::NormalInv)
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&NORMINV_CATALOG_ENTRY];
