use runmat_builtins::{
    XTICKANGLE_DESCRIPTOR, XTICKANGLE_ERROR_TOO_MANY_OUTPUTS, XTICKANGLE_INTEGER_AXES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_angle::axis_tick_angle_builtin;
use super::axis_ticks::TickAxis;

#[runtime_builtin(
    name = "xtickangle",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::xtickangle"
)]
pub fn xtickangle_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    xtickangle_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn xtickangle_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    xtickangle_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn xtickangle_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &XTICKANGLE_DESCRIPTOR,
        &XTICKANGLE_ERROR_TOO_MANY_OUTPUTS,
        "xtickangle",
    )?;
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XTICKANGLE_INTEGER_AXES_EXTENSION,
            "xtickangle",
        )?;
    }
    axis_tick_angle_builtin(runtime, "xtickangle", TickAxis::X, args)
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
    fn xtickangle_sets_queries_and_round_trips_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        assert_eq!(xtickangle_builtin(Vec::new()).unwrap(), Value::Num(0.0));
        assert_eq!(
            xtickangle_builtin(vec![Value::Num(45.0)]).unwrap(),
            Value::Num(45.0)
        );
        assert_eq!(xtickangle_builtin(Vec::new()).unwrap(), Value::Num(45.0));

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("XTickLabelRotation".into())]).unwrap(),
            Value::Num(45.0)
        );
        let x_axis = get_builtin(vec![ax, Value::String("XAxis".into())]).unwrap();
        assert!(xtickangle_builtin(vec![x_axis.clone()]).is_err());
        assert_eq!(
            get_builtin(vec![
                x_axis.clone(),
                Value::String("TickLabelRotation".into())
            ])
            .unwrap(),
            Value::Num(45.0)
        );

        set_builtin(vec![
            x_axis,
            Value::String("TickLabelRotation".into()),
            Value::Num(-30.0),
        ])
        .unwrap();
        assert_eq!(xtickangle_builtin(Vec::new()).unwrap(), Value::Num(-30.0));
    }

    #[test]
    fn xtickangle_axes_target_and_array_setters_are_isolated() {
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

        xtickangle_builtin(vec![Value::Num(ax1), Value::Num(15.0)]).unwrap();
        assert_eq!(
            xtickangle_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::Num(15.0)
        );
        assert_eq!(
            xtickangle_builtin(vec![Value::Num(ax2)]).unwrap(),
            Value::Num(0.0)
        );
        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );

        xtickangle_builtin(vec![tensor(vec![ax1, ax2]), Value::Num(60.0)]).unwrap();
        assert_eq!(
            xtickangle_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::Num(60.0)
        );
        assert_eq!(
            xtickangle_builtin(vec![Value::Num(ax2)]).unwrap(),
            Value::Num(60.0)
        );
        assert!(xtickangle_builtin(vec![tensor(vec![ax1, ax2])]).is_err());
        assert!(xtickangle_builtin(vec![tensor(vec![10.0, 20.0])]).is_err());
        assert!(xtickangle_builtin(vec![Value::Num(f64::NAN)]).is_err());
        assert!(set_builtin(vec![
            Value::Num(ax1),
            Value::String("XTickLabelRotation".into()),
            tensor(vec![10.0, 20.0]),
        ])
        .is_err());
    }
}
