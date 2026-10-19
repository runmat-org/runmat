use runmat_builtins::AXES_CONTROL_SETTER_OUTPUT_EXTENSION;

pub fn validate_setter_output(builtin: &'static str) -> crate::BuiltinResult<()> {
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
            builtin,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::{Tensor, Value};

    fn limits(values: [f64; 2]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, 2]).expect("limits"))
    }

    #[test]
    fn strict_assigned_setter_rejects_before_axes_change_but_query_remains_valid() {
        let _scene = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let baseline = limits([0.0, 2.0]);
        let _zero = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        crate::builtins::plotting::xlim::xlim_builtin(vec![baseline])
            .expect("unassigned setter is MATLAB-compatible");
        drop(_zero);

        let _one = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let error = crate::builtins::plotting::xlim::xlim_builtin(vec![limits([3.0, 5.0])])
            .expect_err("assigned setter is a RunMat-only extension");
        assert_eq!(
            error.identifier(),
            AXES_CONTROL_SETTER_OUTPUT_EXTENSION.error_identifier
        );
        let queried = crate::builtins::plotting::xlim::xlim_builtin(Vec::new())
            .expect("assigned query remains MATLAB-compatible");
        let tensor = Tensor::try_from(&queried).expect("limit vector");
        assert_eq!(tensor.materialize_f64(), vec![0.0, 2.0]);
        drop(_one);

        let _two = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = crate::builtins::plotting::xlim::xlim_builtin(vec![limits([8.0, 9.0])])
            .expect_err("two outputs reject before setting limits");
        assert_eq!(error.identifier(), Some("RunMat:xlim:TooManyOutputs"));
        let queried = crate::builtins::plotting::xlim::xlim_builtin(Vec::new())
            .expect_err("two-output query also rejects before state access");
        assert_eq!(queried.identifier(), Some("RunMat:xlim:TooManyOutputs"));
    }

    #[test]
    fn runmat_assigned_setter_retains_legacy_result() {
        let _scene = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _one = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let result = crate::builtins::plotting::xlim::xlim_builtin(vec![limits([4.0, 7.0])])
            .expect("one-output RunMat setter");
        let tensor = Tensor::try_from(&result).expect("legacy result");
        assert_eq!(tensor.materialize_f64(), vec![4.0, 7.0]);
    }

    #[test]
    fn strict_scale_and_tick_setters_leave_prior_state_unchanged() {
        let _scene = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        {
            let _zero =
                crate::output_count::push_output_count(Some(0)).expect("output-count scope");
            crate::builtins::plotting::xscale::xscale_builtin(vec![Value::String("linear".into())])
                .expect("unassigned scale setter");
            crate::builtins::plotting::xticks::xticks_builtin(vec![limits([1.0, 2.0])])
                .expect("unassigned ticks setter");
        }
        let _one = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        for error in [
            crate::builtins::plotting::xscale::xscale_builtin(vec![Value::String("log".into())])
                .expect_err("assigned scale setter"),
            crate::builtins::plotting::xticks::xticks_builtin(vec![limits([3.0, 4.0])])
                .expect_err("assigned ticks setter"),
        ] {
            assert_eq!(
                error.identifier(),
                AXES_CONTROL_SETTER_OUTPUT_EXTENSION.error_identifier
            );
        }
        assert!(matches!(
            crate::builtins::plotting::xscale::xscale_builtin(Vec::new())
                .expect("scale query"),
            Value::String(scale) if scale == "linear"
        ));
        let ticks =
            crate::builtins::plotting::xticks::xticks_builtin(Vec::new()).expect("tick query");
        assert_eq!(
            Tensor::try_from(&ticks)
                .expect("tick vector")
                .materialize_f64(),
            vec![1.0, 2.0]
        );
    }
}
