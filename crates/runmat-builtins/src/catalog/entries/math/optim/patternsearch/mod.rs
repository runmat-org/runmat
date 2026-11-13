mod descriptor;
mod documentation;
mod nonuniform_documentation;
mod search_documentation;
mod custom_search_documentation;
mod nelder_documentation;
mod genetic_documentation;
mod rbf_documentation;
mod entry;
mod extensions;
mod inference;
mod plot_descriptors;
mod plot_documentation;
mod plot_entries;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub use extensions::*;
pub use plot_descriptors::*;
pub use plot_entries::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &PATTERNSEARCH_CATALOG_ENTRY,
    &PSPLOTBESTF_CATALOG_ENTRY,
    &PSPLOTFUNCOUNT_CATALOG_ENTRY,
    &PSPLOTMESHSIZE_CATALOG_ENTRY,
    &PSPLOTBESTX_CATALOG_ENTRY,
    &PSPLOTMAXCONSTR_CATALOG_ENTRY,
];
