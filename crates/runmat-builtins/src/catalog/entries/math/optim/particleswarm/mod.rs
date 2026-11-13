mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &PARTICLESWARM_CATALOG_ENTRY,
    &PSWCREATIONUNIFORM_CATALOG_ENTRY,
    &PSWPLOTBESTF_CATALOG_ENTRY,
];
