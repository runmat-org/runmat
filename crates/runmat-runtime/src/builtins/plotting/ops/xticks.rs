use runmat_builtins::{
    XTICKS_DESCRIPTOR, XTICKS_ERROR_TOO_MANY_OUTPUTS, XTICKS_INTEGER_AXES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_ticks::{axis_ticks_builtin, TickAxis};

#[runtime_builtin(
    name = "xticks",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::xticks"
)]
pub fn xticks_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    xticks_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn xticks_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    xticks_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn xticks_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &XTICKS_DESCRIPTOR,
        &XTICKS_ERROR_TOO_MANY_OUTPUTS,
        "xticks",
    )?;
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XTICKS_INTEGER_AXES_EXTENSION,
            "xticks",
        )?;
    }
    axis_ticks_builtin(runtime, "xticks", TickAxis::X, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    fn tensor(data: Vec<f64>) -> Value {
        let len = data.len();
        Value::Tensor(runmat_value::Tensor::new(data, vec![1, len]).expect("x tick row"))
    }

    #[test]
    fn scalar_xticks_and_property_controls_share_checked_numeric_normalization() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        xticks_builtin(vec![Value::Num(1.0)]).unwrap();
        assert_eq!(
            runmat_value::Tensor::try_from(&xticks_builtin(Vec::new()).unwrap())
                .unwrap()
                .materialize_f64(),
            vec![1.0]
        );
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        for (value, expected) in [
            (Value::Num(2.0), 2.0),
            (Value::Int(runmat_value::IntValue::I16(3)), 3.0),
        ] {
            crate::builtins::plotting::set::set_builtin(vec![
                ax.clone(),
                Value::from("XTick"),
                value,
            ])
            .unwrap();
            let ticks = get_builtin(vec![ax.clone(), Value::from("XTick")]).unwrap();
            assert_eq!(
                runmat_value::Tensor::try_from(&ticks)
                    .unwrap()
                    .materialize_f64(),
                vec![expected]
            );
            assert_eq!(xticks_builtin(Vec::new()).unwrap(), ticks);
        }
        let before = xticks_builtin(Vec::new()).unwrap();
        assert!(xticks_builtin(vec![Value::Num(f64::INFINITY)]).is_err());
        assert!(crate::builtins::plotting::set::set_builtin(vec![
            ax,
            Value::from("XTick"),
            tensor(vec![1.0, 1.0]),
        ])
        .is_err());
        assert_eq!(xticks_builtin(Vec::new()).unwrap(), before);

        let wide = (1_u64 << 63) + 1;
        let exact = crate::builtins::plotting::axis_ticks::tick_tensor_from_value(
            &Value::Int(runmat_value::IntValue::U64(wide)),
            "xticks",
        )
        .unwrap();
        assert_eq!(
            exact.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![wide]))
        );
    }

    #[test]
    fn xticks_set_query_mode_and_property_round_trip() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let set = xticks_builtin(vec![tensor(vec![0.0, 2.5, 5.0])]).unwrap();
        assert_eq!(
            runmat_value::Tensor::try_from(&set)
                .unwrap()
                .materialize_f64(),
            vec![0.0, 2.5, 5.0]
        );
        assert_eq!(
            xticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );

        let queried = xticks_builtin(Vec::new()).unwrap();
        assert_eq!(
            runmat_value::Tensor::try_from(&queried)
                .unwrap()
                .materialize_f64(),
            vec![0.0, 2.5, 5.0]
        );
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let prop = get_builtin(vec![ax, Value::String("XTick".into())]).unwrap();
        assert_eq!(
            runmat_value::Tensor::try_from(&prop)
                .unwrap()
                .materialize_f64(),
            vec![0.0, 2.5, 5.0]
        );

        let _ = xticks_builtin(vec![Value::String("auto".into())]).unwrap();
        assert_eq!(
            xticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
    }

    #[test]
    fn xticks_axes_target_does_not_change_current_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

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

        let _ = xticks_builtin(vec![Value::Num(ax1), tensor(vec![0.0, 1.0, 2.0])]).unwrap();
        let current = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        assert_eq!(current, Value::Num(ax2));
    }

    #[test]
    fn xticks_auto_query_uses_plotted_data_bounds() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            tensor(vec![10.0, 20.0, 30.0]),
            tensor(vec![1.0, 4.0, 9.0]),
        ]))
        .unwrap();
        let queried = xticks_builtin(Vec::new()).unwrap();
        let ticks = runmat_value::Tensor::try_from(&queried)
            .unwrap()
            .materialize_f64();

        assert_eq!(ticks.first().copied(), Some(10.0));
        assert_eq!(ticks.last().copied(), Some(30.0));
    }
}
