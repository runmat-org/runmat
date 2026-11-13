use runmat_builtins::{
    YTICKS_DESCRIPTOR, YTICKS_ERROR_TOO_MANY_OUTPUTS, YTICKS_INTEGER_AXES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_ticks::{axis_ticks_builtin, TickAxis};

#[runtime_builtin(
    name = "yticks",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::yticks"
)]
pub fn yticks_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    yticks_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn yticks_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    yticks_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn yticks_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &YTICKS_DESCRIPTOR,
        &YTICKS_ERROR_TOO_MANY_OUTPUTS,
        "yticks",
    )?;
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &YTICKS_INTEGER_AXES_EXTENSION,
            "yticks",
        )?;
    }
    axis_ticks_builtin(runtime, "yticks", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    fn tensor(data: Vec<f64>) -> Value {
        let len = data.len();
        Value::Tensor(runmat_value::Tensor::new(data, vec![1, len]).expect("y tick row"))
    }

    #[test]
    fn yticks_supports_properties_and_manual_freeze() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();

        let _ = set_builtin(vec![
            ax.clone(),
            Value::String("YTick".into()),
            tensor(vec![1.0, 3.0, 9.0]),
        ])
        .unwrap();
        let queried = yticks_builtin(Vec::new()).unwrap();
        assert_eq!(
            runmat_value::Tensor::try_from(&queried)
                .unwrap()
                .materialize_f64(),
            vec![1.0, 3.0, 9.0]
        );
        let mode = get_builtin(vec![ax.clone(), Value::String("YTickMode".into())]).unwrap();
        assert_eq!(mode, Value::String("manual".into()));

        let _ = set_builtin(vec![
            ax,
            Value::String("YTickMode".into()),
            Value::String("auto".into()),
        ])
        .unwrap();
        assert_eq!(
            yticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
        let _ = yticks_builtin(vec![Value::String("manual".into())]).unwrap();
        assert_eq!(
            yticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
    }
}
