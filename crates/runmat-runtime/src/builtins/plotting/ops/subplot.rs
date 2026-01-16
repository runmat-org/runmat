//! MATLAB-compatible `subplot` builtin for selecting axes within a figure.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::plotting_error;
use super::state::configure_subplot_with_builtin;

use crate::BuiltinResult;

fn scalar_from_value(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Num(v) => to_positive_index(*v, name),
        Value::Bool(flag) => to_positive_index(if *flag { 1.0 } else { 0.0 }, name),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(plotting_error(name, format!("{name}: expected scalar input")));
            }
            to_positive_index(tensor.data[0], name)
        }
        _ => Err(plotting_error(name, format!("{name}: unsupported argument type"))),
    }
}

fn to_positive_index(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(plotting_error(name, format!("{name}: value must be finite")));
    }
    let rounded = value.round() as i64;
    if rounded <= 0 {
        return Err(plotting_error(name, format!("{name}: value must be positive")));
    }
    Ok(rounded as usize)
}

#[runtime_builtin(
    name = "subplot",
    category = "plotting",
    summary = "Select a subplot grid location.",
    keywords = "subplot,axes,plotting",
    suppress_auto_output = true,
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
