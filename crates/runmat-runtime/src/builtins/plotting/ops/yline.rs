use runmat_builtins::{
    YLINE_DESCRIPTOR, YLINE_ERROR_INTERNAL, YLINE_ERROR_INVALID_ARGUMENT,
    YLINE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ReferenceLineOrientation;
use runmat_value::Value;

use crate::BuiltinResult;

const BUILTIN_NAME: &str = "yline";

#[runtime_builtin(
    name = "yline",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::yline"
)]
pub fn yline_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &YLINE_DESCRIPTOR,
        &YLINE_ERROR_TOO_MANY_OUTPUTS,
        "yline",
    )?;
    super::xline::reference_line_builtin(
        BUILTIN_NAME,
        ReferenceLineOrientation::Horizontal,
        &YLINE_ERROR_INVALID_ARGUMENT,
        &YLINE_ERROR_INTERNAL,
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use runmat_value::Tensor;

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn yline_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = YLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = yline(y)"));
        assert!(labels.contains(&"h = yline(y, styleOrLabel)"));
        assert!(labels.contains(&"h = yline(y, Name, Value, ...)"));
    }

    #[test]
    fn yline_supports_user_repro() {
        let _guard = setup();
        let handle = yline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(0.0),
                Value::String("k".into()),
                Value::String("LineWidth".into()),
                Value::Num(1.0),
            ],
        ))
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
        let err = yline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(
                Tensor::new_2d(vec![0.0, f64::INFINITY], 1, 2).unwrap(),
            )],
        ))
        .expect_err("nonfinite coordinates should fail");
        assert!(err.to_string().contains("finite"));
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::yline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:yline:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::yline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:yline:TooManyOutputs"));
    }
}
