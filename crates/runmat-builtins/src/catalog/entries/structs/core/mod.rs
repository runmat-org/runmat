pub(in crate::catalog) mod structfun;

pub use structfun::{
    STRUCTFUN_CATALOG_ENTRY, STRUCTFUN_DESCRIPTOR, STRUCTFUN_ERROR_FUNCTION_ERROR,
    STRUCTFUN_ERROR_INTERNAL, STRUCTFUN_ERROR_INVALID_INPUT, STRUCTFUN_ERROR_NOT_SCALAR_STRUCT,
    STRUCTFUN_ERROR_UNDEFINED_FUNCTION, STRUCTFUN_ERROR_UNIFORM_OUTPUT,
    STRUCTFUN_INTEGER_CAPABILITIES,
};

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::super::extend_groups(entries, &[structfun::ENTRIES]);
}
