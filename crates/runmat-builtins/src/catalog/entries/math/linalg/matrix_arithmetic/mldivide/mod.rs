mod contract;
mod documentation;

#[cfg(test)]
mod tests;

use crate::*;
pub use contract::*;

pub const MLDIVIDE_CATALOG_ENTRY: BuiltinCatalogEntry = super::contract::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "mldivide",
    documentation::DOCUMENTATION,
    &MLDIVIDE_DESCRIPTOR,
    MatrixArithmeticInferenceRule::LeftDivide,
    MLDIVIDE_INTEGER_CAPABILITIES,
);

pub(super) const fn mldivide_entry() -> &'static BuiltinCatalogEntry {
    &MLDIVIDE_CATALOG_ENTRY
}
