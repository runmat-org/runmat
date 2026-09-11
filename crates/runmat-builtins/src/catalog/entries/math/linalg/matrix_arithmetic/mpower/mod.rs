mod contract;
mod documentation;

#[cfg(test)]
mod tests;

use crate::*;
pub use contract::*;

pub const MPOWER_CATALOG_ENTRY: BuiltinCatalogEntry = super::contract::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "mpower",
    documentation::DOCUMENTATION,
    &MPOWER_DESCRIPTOR,
    MatrixArithmeticInferenceRule::Power,
    MPOWER_INTEGER_CAPABILITIES,
);

pub(super) const fn mpower_entry() -> &'static BuiltinCatalogEntry {
    &MPOWER_CATALOG_ENTRY
}
