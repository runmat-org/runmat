use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ReferenceLineOrientation;

use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "yline";

#[runtime_builtin(
    name = "yline",
    category = "plotting",
    summary = "Draw horizontal reference lines on the current axes.",
    keywords = "yline,reference,line,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::yline"
)]
pub fn yline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::xline::reference_line_builtin(BUILTIN_NAME, ReferenceLineOrientation::Horizontal, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use runmat_builtins::Tensor;

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn yline_supports_user_repro() {
        let _guard = setup();
        let handle = yline_builtin(vec![
            Value::Num(0.0),
            Value::String("k".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.0),
        ])
        .unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Value".into())]).unwrap(),
            Value::Num(0.0)
        );
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 1);
    }

    #[test]
    fn yline_rejects_nonfinite_coordinates() {
        let _guard = setup();
        let err = yline_builtin(vec![Value::Tensor(
            Tensor::new_2d(vec![0.0, f64::INFINITY], 1, 2).unwrap(),
        )])
        .expect_err("nonfinite coordinates should fail");
        assert!(err.to_string().contains("finite"));
    }
}
