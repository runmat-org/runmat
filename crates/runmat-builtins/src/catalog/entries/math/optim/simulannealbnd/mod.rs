mod descriptor;
mod documentation;
mod entry;
mod helper_descriptors;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub use helper_descriptors::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &SIMULANNEALBND_CATALOG_ENTRY,
    &ACCEPTANCESA_CATALOG_ENTRY,
    &ANNEALINGFAST_CATALOG_ENTRY,
    &ANNEALINGBOLTZ_CATALOG_ENTRY,
    &TEMPERATUREEXP_CATALOG_ENTRY,
    &TEMPERATUREFAST_CATALOG_ENTRY,
    &TEMPERATUREBOLTZ_CATALOG_ENTRY,
    &SAPLOTBESTF_CATALOG_ENTRY,
    &SAPLOTBESTX_CATALOG_ENTRY,
    &SAPLOTF_CATALOG_ENTRY,
    &SAPLOTX_CATALOG_ENTRY,
    &SAPLOTSTOPPING_CATALOG_ENTRY,
    &SAPLOTTEMPERATURE_CATALOG_ENTRY,
];
