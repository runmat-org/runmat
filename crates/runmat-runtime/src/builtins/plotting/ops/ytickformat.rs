use runmat_builtins::{YTICKFORMAT_DESCRIPTOR, YTICKFORMAT_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_format::axis_tick_format_builtin;
use super::axis_ticks::TickAxis;

#[runtime_builtin(
    name = "ytickformat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::ytickformat"
)]
pub fn ytickformat_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ytickformat_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn ytickformat_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ytickformat_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn ytickformat_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &YTICKFORMAT_DESCRIPTOR,
        &YTICKFORMAT_ERROR_TOO_MANY_OUTPUTS,
        "ytickformat",
    )?;
    axis_tick_format_builtin(runtime, "ytickformat", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::{label_cell_texts, tensor};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ytickformat_sets_queries_and_formats_auto_labels() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        crate::builtins::plotting::yticks::yticks_builtin(vec![tensor(vec![0.0, 0.5, 1.0])])
            .unwrap();
        assert_eq!(
            ytickformat_builtin(vec![Value::String("percentage".into())]).unwrap(),
            Value::String("%g%%".into())
        );

        let labels =
            crate::builtins::plotting::yticklabels::yticklabels_builtin(Vec::new()).unwrap();
        assert_eq!(label_cell_texts(&labels), vec!["0%", "0.5%", "1%"]);

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let y_axis = get_builtin(vec![ax, Value::String("YAxis".into())]).unwrap();
        let prop = get_builtin(vec![y_axis, Value::String("TickLabelFormat".into())]).unwrap();
        assert_eq!(prop, Value::String("%g%%".into()));
    }
}
