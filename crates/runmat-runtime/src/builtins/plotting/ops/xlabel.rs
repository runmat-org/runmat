use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, parse_text_command};
use super::state::set_xlabel_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

#[runtime_builtin(
    name = "xlabel",
    category = "plotting",
    summary = "Set the current axes x-axis label.",
    keywords = "xlabel,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::xlabel"
)]
pub fn xlabel_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let command = parse_text_command("xlabel", &args)?;
    set_xlabel_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("xlabel", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::ylabel::ylabel_builtin;
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn x_and_y_labels_update_active_axes_metadata() {
        let _guard = setup_plot_tests();
        let xh = xlabel_builtin(vec![Value::String("Time".into())]).unwrap();
        let yh = ylabel_builtin(vec![
            Value::String("Amplitude".into()),
            Value::String("Color".into()),
            Value::String("g".into()),
        ])
        .unwrap();

        assert_eq!(
            decode_plot_object_handle(xh).unwrap().2,
            PlotObjectKind::XLabel
        );
        assert_eq!(
            decode_plot_object_handle(yh).unwrap().2,
            PlotObjectKind::YLabel
        );

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_label.as_deref(), Some("Time"));
        assert_eq!(meta.y_label.as_deref(), Some("Amplitude"));
        assert!(meta.y_label_style.color.is_some());
    }

    #[test]
    fn xlabel_and_ylabel_reject_invalid_property_values() {
        let _guard = setup_plot_tests();

        let err = xlabel_builtin(vec![
            Value::String("Time".into()),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));

        let err = ylabel_builtin(vec![
            Value::String("Amp".into()),
            Value::String("Interpreter".into()),
            Value::Num(5.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("Interpreter must be a string"));
    }
}
