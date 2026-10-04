mod contract;
mod getpref;
mod inference;
mod ispref;
mod setpref;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub use contract::{
    GETPREF_DESCRIPTOR, GETPREF_ERROR_TOO_MANY_OUTPUTS, GETPREF_GROUP_QUERY_EXTENSION,
    GETPREF_INTEGER_CAPABILITIES, ISPREF_DESCRIPTOR, ISPREF_ERROR_TOO_MANY_OUTPUTS,
    ISPREF_INTEGER_AUDIT, SETPREF_DESCRIPTOR, SETPREF_ERROR_TOO_MANY_OUTPUTS,
    SETPREF_INTEGER_CAPABILITIES, SETPREF_STATUS_EXTENSION,
};
pub use getpref::GETPREF_CATALOG_ENTRY;
pub use ispref::ISPREF_CATALOG_ENTRY;
pub use setpref::SETPREF_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &GETPREF_CATALOG_ENTRY,
    &ISPREF_CATALOG_ENTRY,
    &SETPREF_CATALOG_ENTRY,
];

pub(super) fn infer(
    rule: crate::PreferenceInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}
