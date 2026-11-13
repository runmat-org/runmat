mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::INTERP1Q_DESCRIPTOR;
pub use descriptor::{
    ERROR_INTERNAL as INTERP1Q_ERROR_INTERNAL, ERROR_INVALID_INPUT as INTERP1Q_ERROR_INVALID_INPUT,
};
pub use entry::INTERP1Q_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&INTERP1Q_CATALOG_ENTRY];
