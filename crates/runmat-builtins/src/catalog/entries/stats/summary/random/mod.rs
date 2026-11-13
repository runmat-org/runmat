mod capabilities;
mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use capabilities::*;
pub use descriptor::RANDOM_DESCRIPTOR;
pub use entry::RANDOM_CATALOG_ENTRY;

pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::infer(request, entry)
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&RANDOM_CATALOG_ENTRY];
