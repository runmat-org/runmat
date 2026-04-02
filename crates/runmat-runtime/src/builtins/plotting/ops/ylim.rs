use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::limits::{limit_value, parse_limit_command, LimitCommand};
use super::state::{axis_limits_snapshot, set_axis_limits};
use crate::builtins::plotting::type_resolvers::get_type;

#[runtime_builtin(
    name = "ylim",
    category = "plotting",
    summary = "Query or set Y-axis limits.",
    keywords = "ylim,plotting,axes",
    suppress_auto_output = true,
    type_resolver(get_type),
    builtin_path = "crate::builtins::plotting::ylim"
)]
pub fn ylim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match parse_limit_command("ylim", &args)? {
        LimitCommand::Query => Ok(limit_value(axis_limits_snapshot().1)),
        LimitCommand::Set(limits) => {
            let (x, _) = axis_limits_snapshot();
            set_axis_limits(x, limits);
            Ok(limit_value(limits))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ylim_supports_auto_reset() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = ylim_builtin(vec![Value::Tensor(runmat_builtins::Tensor {
            data: vec![1.0, 5.0],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })])
        .unwrap();
        let _ = ylim_builtin(vec![Value::String("auto".into())]).unwrap();
        let queried = ylim_builtin(Vec::new()).unwrap();
        let tensor = runmat_builtins::Tensor::try_from(&queried).unwrap();
        assert!(tensor.data[0].is_nan() && tensor.data[1].is_nan());
    }
}
