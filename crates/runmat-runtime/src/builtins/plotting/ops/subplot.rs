//! MATLAB-compatible `subplot` builtin for selecting axes within a figure.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::scalar_from_value;
use super::state::configure_subplot_with_builtin;
use crate::builtins::plotting::type_resolvers::string_type;

#[runtime_builtin(
    name = "subplot",
    category = "plotting",
    summary = "Select a subplot grid location.",
    keywords = "subplot,axes,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::subplot"
)]
pub fn subplot_builtin(rows: Value, cols: Value, position: Value) -> crate::BuiltinResult<String> {
    let m = scalar_from_value(&rows, "subplot")?;
    let n = scalar_from_value(&cols, "subplot")?;
    let p = scalar_from_value(&position, "subplot")?;
    let zero_based = p.saturating_sub(1);
    configure_subplot_with_builtin("subplot", m, n, zero_based)?;
    Ok(format!("subplot({}, {}, {}) selected", m, n, p))
}
