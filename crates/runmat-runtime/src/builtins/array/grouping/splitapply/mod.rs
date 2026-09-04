//! Grouped function application.

mod assemble;
mod error;
mod execute;
mod extensions;
mod groups;
mod input;
mod invoke;
mod slice;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "splitapply",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::grouping::splitapply"
)]
pub(crate) async fn splitapply_builtin(
    function: Value,
    first_data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    execute::apply(function, first_data, rest).await
}
