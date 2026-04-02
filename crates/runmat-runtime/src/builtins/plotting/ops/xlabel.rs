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
    use runmat_builtins::{CellArray, StringArray};
    use runmat_plot::plots::Figure;

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

    #[test]
    fn xlabel_and_ylabel_support_explicit_axes_targets_and_multiline_text() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 1,
        ));

        let xhandle = xlabel_builtin(vec![
            ax.clone(),
            Value::StringArray(StringArray {
                data: vec!["Time".into(), "(s)".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();
        let yhandle = ylabel_builtin(vec![
            ax,
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("Amplitude".into()),
                        Value::String("(V)".into()),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::String("FontAngle".into()),
            Value::String("italic".into()),
        ])
        .unwrap();

        assert_eq!(decode_plot_object_handle(xhandle).unwrap().1, 1);
        assert_eq!(decode_plot_object_handle(yhandle).unwrap().1, 1);

        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.x_label.as_deref(), Some("Time\n(s)"));
        assert_eq!(meta.y_label.as_deref(), Some("Amplitude\n(V)"));
        assert_eq!(meta.x_label_style.font_weight.as_deref(), Some("bold"));
        assert_eq!(meta.y_label_style.font_angle.as_deref(), Some("italic"));
    }
}
