use runmat_builtins::{
    YTICKLABELS_DESCRIPTOR, YTICKLABELS_ERROR_TOO_MANY_OUTPUTS, YTICKLABELS_INTEGER_AXES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_labels::axis_tick_labels_builtin;
use super::axis_ticks::TickAxis;

#[runtime_builtin(
    name = "yticklabels",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::yticklabels"
)]
pub fn yticklabels_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    yticklabels_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn yticklabels_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    yticklabels_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn yticklabels_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &YTICKLABELS_DESCRIPTOR,
        &YTICKLABELS_ERROR_TOO_MANY_OUTPUTS,
        "yticklabels",
    )?;
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &YTICKLABELS_INTEGER_AXES_EXTENSION,
            "yticklabels",
        )?;
    } else if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        return Err(super::plotting_error(
            "yticklabels",
            "yticklabels: tick labels must be text or categorical values",
        ));
    }
    axis_tick_labels_builtin(runtime, "yticklabels", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::{label_cell_texts, tensor};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::{CellArray, CharArray};

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn yticklabels_supports_set_get_properties_and_manual_freeze() {
        let _guard = setup_plot_tests();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        crate::builtins::plotting::yticks::yticks_builtin(vec![tensor(vec![0.0, 50.0, 100.0])])
            .unwrap();

        let labels = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("low")),
                    Value::CharArray(CharArray::new_row("mid")),
                    Value::CharArray(CharArray::new_row("high")),
                ],
                1,
                3,
            )
            .unwrap(),
        );
        yticklabels_builtin(vec![labels]).unwrap();
        assert_eq!(
            label_cell_texts(&yticklabels_builtin(Vec::new()).unwrap()),
            vec!["low", "mid", "high"]
        );
        assert_eq!(
            yticklabels_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
        let prop = get_builtin(vec![ax.clone(), Value::String("YTickLabel".into())]).unwrap();
        assert_eq!(label_cell_texts(&prop), vec!["low", "mid", "high"]);
        let tick_mode = get_builtin(vec![ax, Value::String("YTickMode".into())]).unwrap();
        assert_eq!(tick_mode, Value::String("manual".into()));
    }

    #[test]
    fn yticklabels_property_set_and_auto_mode_round_trip() {
        let _guard = setup_plot_tests();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        crate::builtins::plotting::yticks::yticks_builtin(vec![tensor(vec![1.0, 4.0])]).unwrap();

        set_builtin(vec![
            ax.clone(),
            Value::String("YTickLabel".into()),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::CharArray(CharArray::new_row("one")),
                        Value::CharArray(CharArray::new_row("four")),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        assert_eq!(
            label_cell_texts(&yticklabels_builtin(Vec::new()).unwrap()),
            vec!["one", "four"]
        );

        set_builtin(vec![
            ax.clone(),
            Value::String("YTickLabelMode".into()),
            Value::String("auto".into()),
        ])
        .unwrap();
        assert_eq!(
            yticklabels_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
        assert_eq!(
            label_cell_texts(&yticklabels_builtin(Vec::new()).unwrap()),
            vec!["1", "4"]
        );
    }
}
