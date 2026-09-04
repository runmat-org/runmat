//! Group counts for array and tabular grouping data.

mod bins;
mod empty_groups;
mod error;
mod execute;
mod extensions;
mod input;
mod options;
mod output;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "groupcounts",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::grouping::groupcounts"
)]
pub(crate) async fn groupcounts_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    execute::apply(first, rest).await
}
