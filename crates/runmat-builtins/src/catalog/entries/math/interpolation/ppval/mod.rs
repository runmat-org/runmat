mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use descriptor::{
    ERROR_INTERNAL as PPVAL_ERROR_INTERNAL, ERROR_INVALID_ARGUMENT as PPVAL_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as PPVAL_ERROR_INVALID_INPUT,
};
pub use entry::PPVAL_CATALOG_ENTRY;
pub use extensions::INTEGER_QUERY_EXTENSION as PPVAL_INTEGER_QUERY_EXTENSION;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PPVAL_CATALOG_ENTRY];
