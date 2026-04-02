use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, parse_text_command};
use super::state::set_zlabel_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

#[runtime_builtin(
    name = "zlabel",
    category = "plotting",
    summary = "Set the current axes z-axis label.",
    keywords = "zlabel,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::zlabel"
)]
pub fn zlabel_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let command = parse_text_command("zlabel", &args)?;
    set_zlabel_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("zlabel", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_builtins::{CellArray, StringArray};

    #[test]
    fn zlabel_sets_axes_local_metadata_and_returns_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = zlabel_builtin(vec![Value::String("Height".into())]).unwrap();
        assert_eq!(
            decode_plot_object_handle(handle).unwrap().2,
            PlotObjectKind::ZLabel
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.z_label.as_deref()),
            Some("Height")
        );
    }

    #[test]
    fn zlabel_supports_axes_target_multiline_and_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            1,
        ));

        let handle = zlabel_builtin(vec![
            ax,
            Value::StringArray(StringArray {
                data: vec!["Height".into(), "(m)".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.z_label.as_deref(), Some("Height\n(m)"));
        assert_eq!(meta.z_label_style.font_weight.as_deref(), Some("bold"));

        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert!(matches!(string, Value::StringArray(_)));
        set_builtin(vec![
            Value::Num(handle),
            Value::String("String".into()),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("Depth".into()), Value::String("(km)".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(1).unwrap().z_label.as_deref(),
            Some("Depth\n(km)")
        );
    }
}
