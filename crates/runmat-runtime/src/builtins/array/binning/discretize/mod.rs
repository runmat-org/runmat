//! Numeric bin assignment for `discretize`.

mod arguments;
mod edges;
mod error;
mod execute;
mod labels;
mod numeric;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "discretize",
    builtin_path = "crate::builtins::array::binning::discretize"
)]
pub(crate) async fn discretize_builtin(
    x: Value,
    edges_or_count: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    execute::apply(x, edges_or_count, rest).await
}
