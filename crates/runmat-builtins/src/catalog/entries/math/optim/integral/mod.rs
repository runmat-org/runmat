mod descriptor;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::INTEGRAL_CATALOG_ENTRY;
pub use options::IntegralOptionName;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&INTEGRAL_CATALOG_ENTRY];
