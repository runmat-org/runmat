use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::properties::{get_properties, resolve_plot_handle};
use crate::builtins::plotting::type_resolvers::get_type;

#[runtime_builtin(
    name = "get",
    category = "plotting",
    summary = "Get properties from plotting handles.",
    keywords = "get,plotting,handle,property",
    suppress_auto_output = true,
    type_resolver(get_type),
    builtin_path = "crate::builtins::plotting::get"
)]
pub fn get_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(crate::builtins::plotting::plotting_error(
            "get",
            "get: expected a plotting handle",
        ));
    }
    let handle = resolve_plot_handle(&args[0], "get")?;
    let property = args
        .get(1)
        .and_then(|v| crate::builtins::plotting::style::value_as_string(v));
    get_properties(handle, property.as_deref(), "get")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::legend::legend_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::title::title_builtin;
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
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
    fn get_reads_text_handle_properties() {
        let _guard = setup();
        let h = title_builtin(vec![
            Value::String("Signal".into()),
            Value::String("FontSize".into()),
            Value::Num(16.0),
        ])
        .unwrap();

        let value = get_builtin(vec![Value::Num(h)]).unwrap();
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        assert_eq!(
            st.fields.get("String"),
            Some(&Value::String("Signal".into()))
        );
        assert_eq!(st.fields.get("FontSize"), Some(&Value::Num(16.0)));

        let string_value =
            get_builtin(vec![Value::Num(h), Value::String("String".into())]).unwrap();
        assert_eq!(string_value, Value::String("Signal".into()));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(h),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();
        let weight = get_builtin(vec![Value::Num(h), Value::String("FontWeight".into())]).unwrap();
        assert_eq!(weight, Value::String("bold".into()));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(h),
            Value::String("String".into()),
            Value::StringArray(runmat_builtins::StringArray {
                data: vec!["Top".into(), "Bottom".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
        ])
        .unwrap();
        let multiline = get_builtin(vec![Value::Num(h), Value::String("String".into())]).unwrap();
        assert!(matches!(multiline, Value::StringArray(_)));
    }

    #[test]
    fn get_reads_axes_and_legend_properties() {
        let _guard = setup();
        let mut figure = Figure::new();
        figure.add_line_plot(
            LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
                .unwrap()
                .with_label("A"),
        );
        let handle = crate::builtins::plotting::import_figure(figure);
        let ax = crate::builtins::plotting::state::encode_axes_handle(handle, 0);
        let legend = legend_builtin(vec![Value::Num(ax)]).unwrap();

        let axes_value = get_builtin(vec![Value::Num(ax)]).unwrap();
        assert!(matches!(axes_value, Value::Struct(_)));
        let title_handle =
            get_builtin(vec![Value::Num(ax), Value::String("Title".into())]).unwrap();
        assert!(matches!(title_handle, Value::Num(_)));

        let legend_value = get_builtin(vec![Value::Num(legend)]).unwrap();
        let Value::Struct(legend_struct) = legend_value else {
            panic!("expected struct");
        };
        assert!(legend_struct.fields.contains_key("String"));
        assert!(legend_struct.fields.contains_key("Visible"));

        let visible =
            get_builtin(vec![Value::Num(ax), Value::String("LegendVisible".into())]).unwrap();
        assert_eq!(visible, Value::Bool(true));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(legend),
            Value::String("Orientation".into()),
            Value::String("horizontal".into()),
        ])
        .unwrap();
        let orientation = get_builtin(vec![
            Value::Num(legend),
            Value::String("Orientation".into()),
        ])
        .unwrap();
        assert_eq!(orientation, Value::String("horizontal".into()));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(legend),
            Value::String("Box".into()),
            Value::Bool(false),
        ])
        .unwrap();
        let box_value = get_builtin(vec![Value::Num(legend), Value::String("Box".into())]).unwrap();
        assert_eq!(box_value, Value::Bool(false));
    }
}
