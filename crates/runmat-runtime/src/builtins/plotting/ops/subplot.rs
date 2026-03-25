//! MATLAB-compatible `subplot` builtin for selecting axes within a figure.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::scalar_from_value;
use super::state::{configure_subplot_with_builtin, current_axes_state, encode_axes_handle};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

#[runtime_builtin(
    name = "subplot",
    category = "plotting",
    summary = "Select a subplot grid location.",
    keywords = "subplot,axes,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::subplot"
)]
pub fn subplot_builtin(rows: Value, cols: Value, position: Value) -> crate::BuiltinResult<f64> {
    let m = scalar_from_value(&rows, "subplot")?;
    let n = scalar_from_value(&cols, "subplot")?;
    let p = scalar_from_value(&position, "subplot")?;
    let zero_based = p.saturating_sub(1);
    configure_subplot_with_builtin("subplot", m, n, zero_based)?;
    let axes = current_axes_state();
    Ok(encode_axes_handle(axes.handle, axes.active_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn subplot_returns_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        assert!(matches!(props, Value::Struct(_)));
    }
}
