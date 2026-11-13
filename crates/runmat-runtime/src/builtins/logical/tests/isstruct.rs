//! Structure-container metadata predicate.

use super::metadata::MetadataBoundary;
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISSTRUCT_CATALOG_ENTRY, ISSTRUCT_ERROR_INTERNAL,
    ISSTRUCT_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISSTRUCT_CATALOG_ENTRY,
    &ISSTRUCT_ERROR_INTERNAL,
    &ISSTRUCT_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::Struct,
);

#[runtime_builtin(
    name = "isstruct",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::isstruct"
)]
async fn isstruct_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}

#[cfg(test)]
#[path = "isstruct/tests.rs"]
mod tests;
