use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::limits::{limit_value, parse_limit_command, LimitCommand};
use super::state::{axis_limits_snapshot, set_axis_limits};
use crate::builtins::plotting::type_resolvers::get_type;

#[runtime_builtin(
    name = "xlim",
    category = "plotting",
    summary = "Query or set X-axis limits.",
    keywords = "xlim,plotting,axes",
    suppress_auto_output = true,
    type_resolver(get_type),
    builtin_path = "crate::builtins::plotting::xlim"
)]
pub fn xlim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match parse_limit_command("xlim", &args)? {
        LimitCommand::Query => Ok(limit_value(axis_limits_snapshot().0)),
        LimitCommand::Set(limits) => {
            let (_, y) = axis_limits_snapshot();
            set_axis_limits(limits, y);
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
    fn xlim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let result = xlim_builtin(vec![Value::Tensor(runmat_builtins::Tensor {
            data: vec![0.0, 10.0],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })])
        .unwrap();
        assert!(matches!(result, Value::Tensor(_)));
        let queried = xlim_builtin(Vec::new()).unwrap();
        let tensor = runmat_builtins::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.data, vec![0.0, 10.0]);
    }
}
