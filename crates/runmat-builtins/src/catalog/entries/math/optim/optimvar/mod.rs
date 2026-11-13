mod properties;
pub use properties::OptimizationVariableProperty;
mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &OPTIMVAR_CATALOG_ENTRY,
    &OPTIMVAR_SUBSREF_CATALOG_ENTRY,
    &OPTIMVAR_SUBSASGN_CATALOG_ENTRY,
];
