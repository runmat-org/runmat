mod contract;
mod documentation;

#[cfg(test)]
mod tests;

use crate::*;

pub use contract::*;

pub const MTIMES_CATALOG_ENTRY: BuiltinCatalogEntry = super::contract::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "mtimes",
    documentation::DOCUMENTATION,
    &MTIMES_DESCRIPTOR,
    MatrixArithmeticInferenceRule::Multiply,
    MTIMES_INTEGER_CAPABILITIES,
);

pub(super) const fn mtimes_entry() -> &'static BuiltinCatalogEntry {
    &MTIMES_CATALOG_ENTRY
}
