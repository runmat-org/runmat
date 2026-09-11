mod contract;
mod documentation;

#[cfg(test)]
mod tests;

use crate::*;
pub use contract::*;

pub const MRDIVIDE_CATALOG_ENTRY: BuiltinCatalogEntry = super::contract::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "mrdivide",
    documentation::DOCUMENTATION,
    &MRDIVIDE_DESCRIPTOR,
    MatrixArithmeticInferenceRule::RightDivide,
    MRDIVIDE_INTEGER_CAPABILITIES,
);

pub(super) const fn mrdivide_entry() -> &'static BuiltinCatalogEntry {
    &MRDIVIDE_CATALOG_ENTRY
}
