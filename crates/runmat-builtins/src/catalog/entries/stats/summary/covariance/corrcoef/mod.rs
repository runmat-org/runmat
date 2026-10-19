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
pub use entry::CORRCOEF_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::CORRCOEF_INTEGER_CAPABILITIES;
pub use options::{CorrcoefOptionKey, CorrcoefRowsOption};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CORRCOEF_CATALOG_ENTRY];
