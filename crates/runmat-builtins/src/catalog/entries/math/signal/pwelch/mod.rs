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
pub use entry::PWELCH_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;
pub use options::{PwelchControlCursor, PwelchControlRole, PwelchKeyword};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PWELCH_CATALOG_ENTRY];
