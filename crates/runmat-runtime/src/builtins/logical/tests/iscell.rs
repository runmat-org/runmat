//! Cell-container metadata predicate.

use super::metadata::MetadataBoundary;
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISCELL_CATALOG_ENTRY, ISCELL_ERROR_INTERNAL, ISCELL_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISCELL_CATALOG_ENTRY,
    &ISCELL_ERROR_INTERNAL,
    &ISCELL_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::Cell,
);

#[runtime_builtin(
    name = "iscell",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::iscell"
)]
async fn iscell_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}

#[cfg(test)]
#[path = "iscell/tests.rs"]
mod tests;
