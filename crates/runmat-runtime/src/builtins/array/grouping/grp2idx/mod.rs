//! Grouping-variable indexing for `grp2idx`.

mod error;
mod execute;
mod input;
mod output;
mod provider;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "grp2idx",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::grouping::grp2idx"
)]
pub(crate) async fn grp2idx_builtin(value: Value) -> BuiltinResult<Value> {
    execute::apply(value).await
}
