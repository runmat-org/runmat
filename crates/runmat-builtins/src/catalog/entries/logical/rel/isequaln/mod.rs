mod documentation;
use super::equality_contract::define_equality;
define_equality!(
    ISEQUALN_CATALOG_ENTRY,
    ISEQUALN_DESCRIPTOR,
    ISEQUALN_ERROR_NOT_ENOUGH_INPUTS,
    ISEQUALN_ERROR_INTERNAL,
    ISEQUALN_ERROR_TOO_MANY_OUTPUTS,
    "isequaln",
    "ISEQUALN",
    documentation::DOCUMENTATION
);
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISEQUALN_CATALOG_ENTRY];
