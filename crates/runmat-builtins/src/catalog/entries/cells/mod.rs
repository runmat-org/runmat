pub(in crate::catalog) mod cellfun;

pub use cellfun::{
    CELLFUN_CATALOG_ENTRY, CELLFUN_DESCRIPTOR, CELLFUN_ERROR_FUNCTION_ERROR,
    CELLFUN_ERROR_INTERNAL, CELLFUN_ERROR_INVALID_INPUT, CELLFUN_ERROR_UNDEFINED_FUNCTION,
    CELLFUN_ERROR_UNIFORM_OUTPUT, CELLFUN_INTEGER_CAPABILITIES,
};

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, &[cellfun::ENTRIES]);
}
