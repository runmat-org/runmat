//! MATLAB-compatible `subplot` builtin for selecting axes within a figure.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::state::configure_subplot;

fn scalar_from_value(value: &Value, name: &str) -> Result<usize, String> {
    match value {
        Value::Num(v) => to_positive_index(*v, name),
        Value::Bool(flag) => to_positive_index(if *flag { 1.0 } else { 0.0 }, name),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(format!("{name}: expected scalar input"));
            }
            to_positive_index(tensor.data[0], name)
        }
        _ => Err(format!("{name}: unsupported argument type")),
    }
}

fn to_positive_index(value: f64, name: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!("{name}: value must be finite"));
    }
    let rounded = value.round() as i64;
    if rounded <= 0 {
        return Err(format!("{name}: value must be positive"));
    }
    Ok(rounded as usize)
}

#[runtime_builtin(
    name = "subplot",
    category = "plotting",
    summary = "Select a subplot grid location.",
    keywords = "subplot,axes,plotting"
)]
pub fn subplot_builtin(rows: Value, cols: Value, position: Value) -> Result<String, String> {
    let m = scalar_from_value(&rows, "subplot")?;
    let n = scalar_from_value(&cols, "subplot")?;
    let p = scalar_from_value(&position, "subplot")?;
    let zero_based = p.saturating_sub(1);
    configure_subplot(m, n, zero_based);
    Ok(format!("subplot({}, {}, {}) selected", m, n, p))
}
