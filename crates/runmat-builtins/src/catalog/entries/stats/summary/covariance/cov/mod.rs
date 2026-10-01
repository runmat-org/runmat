mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
mod options;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use entry::COV_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::COV_INTEGER_CAPABILITIES;
pub use options::CovRowsOption;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&COV_CATALOG_ENTRY];
