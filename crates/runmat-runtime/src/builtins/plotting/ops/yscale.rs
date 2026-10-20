use runmat_builtins::{YSCALE_DESCRIPTOR, YSCALE_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_scale::axis_scale_builtin;
use super::axis_ticks::TickAxis;

#[runtime_builtin(
    name = "yscale",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::yscale"
)]
pub fn yscale_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    yscale_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn yscale_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    yscale_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn yscale_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &YSCALE_DESCRIPTOR,
        &YSCALE_ERROR_TOO_MANY_OUTPUTS,
        "yscale",
    )?;
    axis_scale_builtin(runtime, "yscale", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn yscale_queries_sets_and_round_trips_property() {
        let _guard = setup_plot_tests();
        assert_eq!(
            yscale_builtin(Vec::new()).unwrap(),
            Value::String("linear".into())
        );
        assert_eq!(
            yscale_builtin(vec![Value::String("log".into())]).unwrap(),
            Value::String("log".into())
        );
        assert_eq!(
            yscale_builtin(Vec::new()).unwrap(),
            Value::String("log".into())
        );

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let prop = get_builtin(vec![ax.clone(), Value::String("YScale".into())]).unwrap();
        assert_eq!(prop, Value::String("log".into()));

        set_builtin(vec![
            ax,
            Value::String("YScale".into()),
            Value::String("linear".into()),
        ])
        .unwrap();
        assert_eq!(
            yscale_builtin(Vec::new()).unwrap(),
            Value::String("linear".into())
        );
    }

    #[test]
    fn yscale_explicit_axes_does_not_change_current_axes() {
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

        yscale_builtin(vec![Value::Num(ax1), Value::String("log".into())]).unwrap();
        assert_eq!(
            yscale_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::String("log".into())
        );
        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );

        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(fig.axes_metadata(0).unwrap().y_log);
        assert!(!fig.axes_metadata(1).unwrap().y_log);
    }

    #[test]
    fn yscale_rejects_unknown_scale() {
        let _guard = setup_plot_tests();
        let err = yscale_builtin(vec![Value::String("sqrt".into())]).unwrap_err();
        assert!(err.message.contains("scale must be 'linear' or 'log'"));
    }

    #[test]
    fn yscale_rejects_invalid_numeric_axes_handle() {
        let _guard = setup_plot_tests();
        let err =
            yscale_builtin(vec![Value::Num(9_999_000.0), Value::String("log".into())]).unwrap_err();
        assert!(err.message.contains("invalid axes handle"));
    }

    #[test]
    fn yscale_descriptor_only_advertises_query_output() {
        let signatures = YSCALE_DESCRIPTOR.signatures;
        assert_eq!(
            signatures.iter().map(|signature| signature.outputs.len()).collect::<Vec<_>>(),
            [1, 1, 0, 0],
            "both current/explicit-axes queries return a value; neither setter advertises an output",
        );
    }
}
