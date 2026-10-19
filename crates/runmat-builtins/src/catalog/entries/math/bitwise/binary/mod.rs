mod bitand;
mod bitor;
mod bitxor;
mod support;

pub use bitand::*;
pub use bitor::*;
pub use bitxor::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.push(&BITAND_CATALOG_ENTRY);
    values.push(&BITOR_CATALOG_ENTRY);
    values.push(&BITXOR_CATALOG_ENTRY);
}
