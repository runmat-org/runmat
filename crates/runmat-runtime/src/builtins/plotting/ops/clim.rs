use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::limits::{limit_value, parse_limit_command, LimitCommand};
use super::state::{color_limits_snapshot, set_color_limits_runtime};
use crate::builtins::plotting::type_resolvers::get_type;

#[runtime_builtin(
    name = "clim",
    category = "plotting",
    summary = "Query or set color limits.",
    keywords = "clim,plotting,color",
    suppress_auto_output = true,
    type_resolver(get_type),
    builtin_path = "crate::builtins::plotting::clim"
)]
pub fn clim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match parse_limit_command("clim", &args)? {
        LimitCommand::Query => Ok(limit_value(color_limits_snapshot())),
        LimitCommand::Set(limits) => {
            set_color_limits_runtime(limits);
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
    fn clim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = clim_builtin(vec![Value::Tensor(runmat_builtins::Tensor {
            data: vec![0.25, 0.75],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })])
        .unwrap();
        let queried = clim_builtin(Vec::new()).unwrap();
        let tensor = runmat_builtins::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.data, vec![0.25, 0.75]);
    }
}
