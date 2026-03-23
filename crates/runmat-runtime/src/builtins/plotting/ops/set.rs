use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::properties::{resolve_plot_handle, set_properties};
use crate::builtins::plotting::type_resolvers::set_type;

#[runtime_builtin(
    name = "set",
    category = "plotting",
    summary = "Set properties on plotting handles.",
    keywords = "set,plotting,handle,property",
    suppress_auto_output = true,
    type_resolver(set_type),
    builtin_path = "crate::builtins::plotting::set"
)]
pub fn set_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    if args.len() < 3 {
        return Err(crate::builtins::plotting::plotting_error(
            "set",
            "set: expected a plotting handle followed by property/value pairs",
        ));
    }
    let handle = resolve_plot_handle(&args[0], "set")?;
    set_properties(handle, &args[1..], "set")?;
    Ok("ok".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::legend::legend_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::title::title_builtin;
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use runmat_builtins::Value;
    use runmat_plot::plots::{Figure, LinePlot};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn set_updates_text_handle_properties() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        set_builtin(vec![
            Value::Num(h),
            Value::String("String".into()),
            Value::String("Updated".into()),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let title = get_builtin(vec![Value::Num(h), Value::String("String".into())]).unwrap();
        assert_eq!(title, Value::String("Updated".into()));
        let visible = get_builtin(vec![Value::Num(h), Value::String("Visible".into())]).unwrap();
        assert_eq!(visible, Value::Bool(false));
    }

    #[test]
    fn set_updates_extended_text_and_legend_properties() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        set_builtin(vec![
            Value::Num(h),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
            Value::String("FontAngle".into()),
            Value::String("italic".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![Value::Num(h), Value::String("FontWeight".into())]).unwrap(),
            Value::String("bold".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(h), Value::String("FontAngle".into())]).unwrap(),
            Value::String("italic".into())
        );

        let mut figure = Figure::new();
        figure.add_line_plot(
            LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
                .unwrap()
                .with_label("A"),
        );
        let figure = crate::builtins::plotting::import_figure(figure);
        let ax = crate::builtins::plotting::state::encode_axes_handle(figure, 0);
        let legend = legend_builtin(vec![Value::Num(ax)]).unwrap();
        set_builtin(vec![
            Value::Num(legend),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
            Value::String("FontAngle".into()),
            Value::String("italic".into()),
            Value::String("Interpreter".into()),
            Value::String("none".into()),
            Value::String("Box".into()),
            Value::Bool(false),
            Value::String("Orientation".into()),
            Value::String("horizontal".into()),
        ])
        .unwrap();
        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.legend_style.font_weight.as_deref(), Some("bold"));
        assert_eq!(meta.legend_style.font_angle.as_deref(), Some("italic"));
        assert_eq!(meta.legend_style.interpreter.as_deref(), Some("none"));
        assert_eq!(meta.legend_style.box_visible, Some(false));
        assert_eq!(meta.legend_style.orientation.as_deref(), Some("horizontal"));
    }

    #[test]
    fn set_updates_legend_handle_properties() {
        let _guard = setup();
        let mut figure = Figure::new();
        figure.add_line_plot(
            LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
                .unwrap()
                .with_label("A"),
        );
        let figure = crate::builtins::plotting::import_figure(figure);
        let ax = crate::builtins::plotting::state::encode_axes_handle(figure, 0);
        let legend = legend_builtin(vec![Value::Num(ax)]).unwrap();

        set_builtin(vec![
            Value::Num(legend),
            Value::String("Location".into()),
            Value::String("southwest".into()),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.legend_style.location.as_deref(), Some("southwest"));
        assert!(!meta.legend_enabled);
    }

    #[test]
    fn set_updates_axes_handle_alias_properties() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        let (figure, axes, _) =
            crate::builtins::plotting::state::decode_plot_object_handle(h).unwrap();
        let ax = crate::builtins::plotting::state::encode_axes_handle(figure, axes);

        set_builtin(vec![
            Value::Num(ax),
            Value::String("Title".into()),
            Value::String("Updated Title".into()),
            Value::String("LegendVisible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(axes).unwrap();
        assert_eq!(meta.title.as_deref(), Some("Updated Title"));
        assert!(!meta.legend_enabled);
    }

    #[test]
    fn set_rejects_invalid_property_assignments() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        let err = set_builtin(vec![
            Value::Num(h),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));
    }
}
