//! Sorted group discovery and identifier projection.

mod error;
mod execute;
mod extensions;
mod input;
mod output;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "findgroups",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::grouping::findgroups"
)]
pub(crate) async fn findgroups_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    execute::apply(first, rest).await
}
