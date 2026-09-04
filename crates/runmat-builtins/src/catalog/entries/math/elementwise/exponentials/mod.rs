mod exp;
mod expm1;

pub use exp::*;
pub use expm1::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&EXP_CATALOG_ENTRY, &EXPM1_CATALOG_ENTRY];
