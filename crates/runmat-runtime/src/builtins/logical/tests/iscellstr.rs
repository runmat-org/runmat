//! Cell-array-of-character-arrays metadata predicate.

use super::metadata::MetadataBoundary;
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISCELLSTR_CATALOG_ENTRY, ISCELLSTR_ERROR_INTERNAL,
    ISCELLSTR_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISCELLSTR_CATALOG_ENTRY,
    &ISCELLSTR_ERROR_INTERNAL,
    &ISCELLSTR_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::CellString,
);

#[runtime_builtin(
    name = "iscellstr",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::iscellstr"
)]
async fn iscellstr_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}

#[cfg(test)]
#[path = "iscellstr/tests.rs"]
mod tests;
