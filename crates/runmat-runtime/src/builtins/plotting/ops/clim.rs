use runmat_builtins::{CLIM_DESCRIPTOR, CLIM_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::color_limits::apply_color_limits;

#[runtime_builtin(
    name = "clim",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::clim"
)]
pub fn clim_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    clim_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn clim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    clim_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn clim_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &CLIM_DESCRIPTOR,
        &CLIM_ERROR_TOO_MANY_OUTPUTS,
        "clim",
    )?;
    apply_color_limits(runtime, "clim", args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn clim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = clim()"));
        assert!(labels.contains(&"clim([cmin cmax])"));
        assert!(labels.contains(&"clim('auto'|'manual')"));
    }

    #[test]
    fn clim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = clim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![0.25, 0.75], vec![1, 2]).expect("color limits"),
        )])
        .unwrap();
        let queried = clim_builtin(Vec::new()).unwrap();
        let tensor = runmat_value::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![0.25, 0.75]);
    }

    #[test]
    fn named_axes_targets_preserve_current_selection_and_displayed_property() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_plot_state();
        super::super::state::configure_subplot(1, 2, 0).expect("first axes");
        let first = super::super::state::current_axes_state();
        let target = Value::Num(super::super::state::encode_axes_handle(first.handle, 0));
        super::super::state::configure_subplot(1, 2, 1).expect("second axes");
        let active_before = super::super::state::current_axes_state().active_index;

        struct TargetLimitCase {
            property_name: &'static str,
            builtin: fn(Vec<Value>) -> crate::BuiltinResult<Value>,
            expected: [f64; 2],
            output_error: &'static runmat_builtins::BuiltinErrorDescriptor,
        }
        let cases = [
            TargetLimitCase {
                property_name: "XLim",
                builtin: super::super::xlim::xlim_builtin,
                expected: [2.0, 8.0],
                output_error: &runmat_builtins::XLIM_ERROR_TOO_MANY_OUTPUTS,
            },
            TargetLimitCase {
                property_name: "YLim",
                builtin: super::super::ylim::ylim_builtin,
                expected: [-2.0, 4.0],
                output_error: &runmat_builtins::YLIM_ERROR_TOO_MANY_OUTPUTS,
            },
            TargetLimitCase {
                property_name: "ZLim",
                builtin: super::super::zlim::zlim_builtin,
                expected: [-4.0, 4.0],
                output_error: &runmat_builtins::ZLIM_ERROR_TOO_MANY_OUTPUTS,
            },
            TargetLimitCase {
                property_name: "CLim",
                builtin: clim_builtin,
                expected: [0.25, 0.75],
                output_error: &runmat_builtins::CLIM_ERROR_TOO_MANY_OUTPUTS,
            },
            TargetLimitCase {
                property_name: "CLim",
                builtin: super::super::caxis::caxis_builtin,
                expected: [0.1, 0.9],
                output_error: &runmat_builtins::CAXIS_ERROR_TOO_MANY_OUTPUTS,
            },
        ];
        for TargetLimitCase {
            property_name,
            builtin,
            expected,
            output_error,
        } in cases
        {
            let limits = Value::Tensor(
                runmat_value::Tensor::new(expected.to_vec(), vec![1, 2]).expect("limits"),
            );
            builtin(vec![target.clone(), limits]).expect("target setter");
            {
                let _outputs =
                    crate::output_count::push_output_count(Some(2)).expect("output-count scope");
                let rejected = builtin(vec![
                    target.clone(),
                    Value::Tensor(runmat_value::Tensor::new(vec![91.0, 92.0], vec![1, 2]).unwrap()),
                ])
                .expect_err("invalid requested outputs reject before changing the target");
                assert_eq!(rejected.identifier(), output_error.identifier);
            }
            assert_eq!(
                super::super::state::current_axes_state().active_index,
                active_before
            );
            let queried = builtin(vec![target.clone()]).expect("target query");
            assert_eq!(
                runmat_value::Tensor::try_from(&queried)
                    .expect("limits tensor")
                    .materialize_f64(),
                expected.to_vec()
            );
            let property = super::super::properties::get_properties(
                super::super::properties::PlotHandle::Axes(first.handle, 0),
                Some(property_name),
                "get",
            )
            .expect("target property");
            assert_eq!(
                runmat_value::Tensor::try_from(&property)
                    .expect("property tensor")
                    .materialize_f64(),
                expected.to_vec()
            );
        }

        let mut prior_limits = vec![0.1, 0.9];
        for builtin in [clim_builtin, super::super::caxis::caxis_builtin] {
            let integer_limits = || {
                Value::Tensor(
                    runmat_value::Tensor::new_integer(
                        runmat_value::IntegerStorage::U8(vec![1, 7]),
                        vec![1, 2],
                    )
                    .unwrap(),
                )
            };
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let _outputs =
                    crate::output_count::push_output_count(Some(0)).expect("output-count scope");
                let error = builtin(vec![target.clone(), integer_limits()])
                    .expect_err("explicit target does not hide the integer control extension");
                assert_eq!(
                    error.identifier(),
                    runmat_builtins::CLIM_INTEGER_LIMITS_EXTENSION.error_identifier
                );
            }
            let queried = builtin(vec![target.clone()]).expect("prior target remains readable");
            assert_eq!(
                runmat_value::Tensor::try_from(&queried)
                    .unwrap()
                    .materialize_f64(),
                prior_limits
            );
            {
                let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
                let _outputs =
                    crate::output_count::push_output_count(Some(0)).expect("output-count scope");
                builtin(vec![target.clone(), integer_limits()])
                    .expect("RunMat integer target setter");
            }
            assert_eq!(
                super::super::state::current_axes_state().active_index,
                active_before
            );
            let queried = builtin(vec![target.clone()]).unwrap();
            assert_eq!(
                runmat_value::Tensor::try_from(&queried)
                    .unwrap()
                    .materialize_f64(),
                vec![1.0, 7.0]
            );
            prior_limits = vec![1.0, 7.0];
        }
    }
}
