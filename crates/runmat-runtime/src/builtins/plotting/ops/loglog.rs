use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::plot::plot_builtin;
use super::state::{current_axes_state, set_log_modes_for_axes};
use crate::builtins::plotting::type_resolvers::string_type;

#[runtime_builtin(
    name = "loglog",
    category = "plotting",
    summary = "Create a plot with logarithmic X and Y axes.",
    keywords = "loglog,plotting,log",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::loglog"
)]
pub async fn loglog_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let result = plot_builtin(args).await;
    let axes = current_axes_state();
    set_log_modes_for_axes(axes.handle, axes.active_index, true, true).map_err(|err| {
        crate::builtins::plotting::plotting_error_with_source(
            "loglog",
            format!("loglog: {err}"),
            err,
        )
    })?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use runmat_builtins::{NumericDType, Tensor};

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: NumericDType::F64,
        }
    }

    #[test]
    fn loglog_sets_both_axes_log_on_active_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(loglog_builtin(vec![Value::Tensor(tensor_from(&[
            1.0, 10.0, 100.0,
        ]))]));
        let fig = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert!(meta.x_log);
        assert!(meta.y_log);
    }
}
