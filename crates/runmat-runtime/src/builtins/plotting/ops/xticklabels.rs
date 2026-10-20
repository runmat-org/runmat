use runmat_builtins::{
    XTICKLABELS_DESCRIPTOR, XTICKLABELS_ERROR_TOO_MANY_OUTPUTS, XTICKLABELS_INTEGER_AXES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_labels::axis_tick_labels_builtin;
use super::axis_ticks::TickAxis;

#[runtime_builtin(
    name = "xticklabels",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::xticklabels"
)]
pub fn xticklabels_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    xticklabels_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn xticklabels_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    xticklabels_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn xticklabels_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &XTICKLABELS_DESCRIPTOR,
        &XTICKLABELS_ERROR_TOO_MANY_OUTPUTS,
        "xticklabels",
    )?;
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XTICKLABELS_INTEGER_AXES_EXTENSION,
            "xticklabels",
        )?;
    } else if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        return Err(super::plotting_error(
            "xticklabels",
            "xticklabels: tick labels must be text or categorical values",
        ));
    }
    axis_tick_labels_builtin(runtime, "xticklabels", TickAxis::X, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::{label_cell_texts, tensor};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::{CellArray, CharArray, StringArray};

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn xticklabels_set_query_mode_and_property_round_trip() {
        let _guard = setup_plot_tests();
        crate::builtins::plotting::xticks::xticks_builtin(vec![tensor(vec![0.0, 5.0, 10.0])])
            .unwrap();
        let labels = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("zero")),
                    Value::CharArray(CharArray::new_row("five")),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let set = xticklabels_builtin(vec![labels]).unwrap();
        assert_eq!(label_cell_texts(&set), vec!["zero", "five", ""]);
        assert_eq!(
            xticklabels_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );

        let queried = xticklabels_builtin(Vec::new()).unwrap();
        assert_eq!(label_cell_texts(&queried), vec!["zero", "five", ""]);
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let prop = get_builtin(vec![ax.clone(), Value::String("XTickLabel".into())]).unwrap();
        assert_eq!(label_cell_texts(&prop), vec!["zero", "five", ""]);
        let mode = get_builtin(vec![ax, Value::String("XTickLabelMode".into())]).unwrap();
        assert_eq!(mode, Value::String("manual".into()));
        for text in ["auto", "manual", "mode"] {
            let labels =
                Value::StringArray(StringArray::new(vec![text.into()], vec![1, 1]).unwrap());
            let result = xticklabels_builtin(vec![labels]).unwrap();
            assert_eq!(label_cell_texts(&result), vec![text, "", ""]);
            assert_eq!(
                xticklabels_builtin(vec![Value::String("mode".into())]).unwrap(),
                Value::String("manual".into()),
                "singleton StringArray remains labels, not a mode token",
            );
        }
    }

    #[test]
    fn xticklabels_auto_generates_from_ticks_and_axes_target_isolated() {
        let _guard = setup_plot_tests();
        let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(1.0),
        )
        .unwrap();
        let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        crate::builtins::plotting::xticks::xticks_builtin(vec![
            Value::Num(ax1),
            tensor(vec![1.0, 2.5]),
        ])
        .unwrap();

        let labels = Value::StringArray(
            StringArray::new(vec!["one".into(), "two point five".into()], vec![1, 2]).unwrap(),
        );
        xticklabels_builtin(vec![Value::Num(ax1), labels]).unwrap();
        assert_eq!(
            xticklabels_builtin(vec![Value::Num(ax1), Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
        assert_eq!(
            label_cell_texts(&xticklabels_builtin(vec![Value::Num(ax1)]).unwrap()),
            vec!["one", "two point five"]
        );
        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );

        xticklabels_builtin(vec![Value::Num(ax1), Value::String("auto".into())]).unwrap();
        assert_eq!(
            xticklabels_builtin(vec![Value::Num(ax1), Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
        assert_eq!(
            label_cell_texts(&xticklabels_builtin(vec![Value::Num(ax1)]).unwrap()),
            vec!["1", "2.5"]
        );
    }
}
