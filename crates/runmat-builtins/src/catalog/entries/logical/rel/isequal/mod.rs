mod documentation;
use super::equality_contract::define_equality;
define_equality!(
    ISEQUAL_CATALOG_ENTRY,
    ISEQUAL_DESCRIPTOR,
    ISEQUAL_ERROR_NOT_ENOUGH_INPUTS,
    ISEQUAL_ERROR_INTERNAL,
    ISEQUAL_ERROR_TOO_MANY_OUTPUTS,
    "isequal",
    "ISEQUAL",
    documentation::DOCUMENTATION
);
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISEQUAL_CATALOG_ENTRY];
