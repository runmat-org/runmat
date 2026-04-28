use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::properties::{get_properties, resolve_plot_handle};
use crate::builtins::plotting::type_resolvers::get_type;

#[runtime_builtin(
    name = "get",
    category = "plotting",
    summary = "Get properties from plotting handles.",
    keywords = "get,plotting,handle,property",
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
    use crate::builtins::plotting::isgraphics::isgraphics_builtin;
    use crate::builtins::plotting::ishandle::ishandle_builtin;
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

    #[test]
    fn get_reads_axes_local_limit_and_scale_properties() {
        let _guard = setup();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(ax),
            Value::String("XLim".into()),
            Value::Tensor(runmat_builtins::Tensor {
                rows: 1,
                cols: 2,
                shape: vec![1, 2],
                data: vec![1.0, 5.0],
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Value::String("XScale".into()),
            Value::String("log".into()),
            Value::String("Grid".into()),
            Value::Bool(false),
        ])
        .unwrap();
        let xlim = get_builtin(vec![Value::Num(ax), Value::String("XLim".into())]).unwrap();
        let scale = get_builtin(vec![Value::Num(ax), Value::String("XScale".into())]).unwrap();
        let grid = get_builtin(vec![Value::Num(ax), Value::String("Grid".into())]).unwrap();
        assert_eq!(
            runmat_builtins::Tensor::try_from(&xlim).unwrap().data,
            vec![1.0, 5.0]
        );
        assert_eq!(scale, Value::String("log".into()));
        assert_eq!(grid, Value::Bool(false));
    }

    #[test]
    fn get_reads_figure_properties() {
        let _guard = setup();
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(1234.0)]).unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let value = get_builtin(vec![Value::Num(fig)]).unwrap();
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        assert_eq!(st.fields.get("Number"), Some(&Value::Num(1234.0)));
        assert_eq!(st.fields.get("Type"), Some(&Value::String("figure".into())));
        assert_eq!(st.fields.get("CurrentAxes"), Some(&Value::Num(ax)));
        assert!(matches!(st.fields.get("Children"), Some(Value::Tensor(_))));
        let parent = st.fields.get("Parent").expect("parent property");
        let Value::Num(v) = parent else {
            panic!("expected numeric parent");
        };
        assert!(v.is_nan());
        let sgtitle = get_builtin(vec![Value::Num(fig), Value::String("SGTitle".into())]).unwrap();
        assert!(matches!(sgtitle, Value::Num(_)));
    }

    #[test]
    fn hierarchy_and_query_semantics_exist() {
        let _guard = setup();
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(5555.0)]).unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .unwrap();
        let title =
            crate::builtins::plotting::title::title_builtin(vec![Value::String("Signal".into())])
                .unwrap();

        let axes_value = get_builtin(vec![Value::Num(ax)]).unwrap();
        let Value::Struct(axes_struct) = axes_value else {
            panic!("expected struct");
        };
        assert_eq!(
            axes_struct.fields.get("Type"),
            Some(&Value::String("axes".into()))
        );
        assert_eq!(axes_struct.fields.get("Parent"), Some(&Value::Num(fig)));
        assert!(matches!(
            axes_struct.fields.get("Children"),
            Some(Value::Tensor(_))
        ));

        let text_value = get_builtin(vec![Value::Num(title)]).unwrap();
        let Value::Struct(text_struct) = text_value else {
            panic!("expected struct");
        };
        assert_eq!(
            text_struct.fields.get("Type"),
            Some(&Value::String("text".into()))
        );
        assert_eq!(text_struct.fields.get("Parent"), Some(&Value::Num(ax)));

        assert!(ishandle_builtin(vec![Value::Num(fig)]).unwrap());
        assert!(isgraphics_builtin(vec![Value::Num(ax)]).unwrap());
        assert!(!ishandle_builtin(vec![Value::Num(-1.0)]).unwrap());
    }

    #[test]
    fn stem_handle_exposes_runtime_properties() {
        let _guard = setup();
        let handle = crate::builtins::plotting::stem::stem_builtin(vec![
            Value::Tensor(runmat_builtins::Tensor {
                rows: 2,
                cols: 1,
                shape: vec![2],
                data: vec![1.0, 2.0],
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Value::String("DisplayName".into()),
            Value::String("Impulse".into()),
            Value::String("BaseValue".into()),
            Value::Num(-1.0),
            Value::String("filled".into()),
        ])
        .unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        let Value::Struct(st) = props else {
            panic!("expected struct");
        };
        assert_eq!(st.fields.get("Type"), Some(&Value::String("stem".into())));
        assert_eq!(st.fields.get("BaseValue"), Some(&Value::Num(-1.0)));
        assert_eq!(st.fields.get("Filled"), Some(&Value::Bool(true)));
    }
}
