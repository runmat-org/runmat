use runmat_builtins::{
    YTICKANGLE_DESCRIPTOR, YTICKANGLE_ERROR_TOO_MANY_OUTPUTS, YTICKANGLE_INTEGER_AXES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_angle::axis_tick_angle_builtin;
use super::axis_ticks::TickAxis;

#[runtime_builtin(
    name = "ytickangle",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::ytickangle"
)]
pub fn ytickangle_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ytickangle_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn ytickangle_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ytickangle_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn ytickangle_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &YTICKANGLE_DESCRIPTOR,
        &YTICKANGLE_ERROR_TOO_MANY_OUTPUTS,
        "ytickangle",
    )?;
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &YTICKANGLE_INTEGER_AXES_EXTENSION,
            "ytickangle",
        )?;
    }
    axis_tick_angle_builtin(runtime, "ytickangle", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::tensor;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ytickangle_sets_queries_and_round_trips_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        assert_eq!(ytickangle_builtin(Vec::new()).unwrap(), Value::Num(0.0));
        assert_eq!(
            ytickangle_builtin(vec![Value::Num(25.0)]).unwrap(),
            Value::Num(25.0)
        );
        assert_eq!(ytickangle_builtin(Vec::new()).unwrap(), Value::Num(25.0));

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("YTickLabelRotation".into())]).unwrap(),
            Value::Num(25.0)
        );
        let y_axis = get_builtin(vec![ax, Value::String("YAxis".into())]).unwrap();
        assert!(ytickangle_builtin(vec![y_axis.clone()]).is_err());
        assert_eq!(
            get_builtin(vec![
                y_axis.clone(),
                Value::String("TickLabelRotation".into())
            ])
            .unwrap(),
            Value::Num(25.0)
        );

        set_builtin(vec![
            y_axis,
            Value::String("TickLabelRotation".into()),
            Value::Num(-45.0),
        ])
        .unwrap();
        assert_eq!(ytickangle_builtin(Vec::new()).unwrap(), Value::Num(-45.0));
    }

    #[test]
    fn ytickangle_axes_array_sets_each_axes_and_rejects_array_query() {
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

        ytickangle_builtin(vec![tensor(vec![ax1, ax2]), Value::Num(35.0)]).unwrap();
        assert_eq!(
            ytickangle_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::Num(35.0)
        );
        assert_eq!(
            ytickangle_builtin(vec![Value::Num(ax2)]).unwrap(),
            Value::Num(35.0)
        );
        assert!(ytickangle_builtin(vec![tensor(vec![ax1, ax2])]).is_err());
        assert!(ytickangle_builtin(vec![tensor(vec![10.0, 20.0])]).is_err());
        assert!(ytickangle_builtin(vec![Value::Num(f64::INFINITY)]).is_err());
        assert!(set_builtin(vec![
            Value::Num(ax1),
            Value::String("YTickLabelRotation".into()),
            tensor(vec![10.0, 20.0]),
        ])
        .is_err());
    }
}
