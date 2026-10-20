use runmat_builtins::{YLIM_DESCRIPTOR, YLIM_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::figure::AxisLimitAxis;
use runmat_value::Value;

use super::op_common::limits::{
    limit_state_error, limit_value, prepare_limit_call, query_limit_method, set_limit_method,
    LimitCommand,
};
use super::state::{
    axes_metadata_snapshot_for_property, axis_display_bounds_snapshot_for_axes_for_property,
    axis_displayed_limits_snapshot_for_axes_for_property,
    set_axis_manual_limits_for_axes_for_property,
};

#[runtime_builtin(
    name = "ylim",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::ylim"
)]
pub fn ylim_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ylim_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn ylim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ylim_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn ylim_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &YLIM_DESCRIPTOR,
        &YLIM_ERROR_TOO_MANY_OUTPUTS,
        "ylim",
    )?;
    let call = prepare_limit_call(runtime, "ylim", args.view(), false)?;

    if matches!(
        call.command,
        LimitCommand::SetMethod(_) | LimitCommand::Freeze | LimitCommand::Set(_)
    ) {
        super::op_common::axes_output::validate_setter_output("ylim")?;
    }
    let target = call.target.resolve("ylim")?;
    let (handle, index) = target.coordinates();
    let property = target.authority();
    match call.command {
        LimitCommand::Query => {
            let limits = axis_displayed_limits_snapshot_for_axes_for_property(
                property,
                handle,
                index,
                AxisLimitAxis::Y,
            )
            .map_err(|err| limit_state_error("ylim", err))?;
            Ok(limit_value(limits))
        }
        LimitCommand::QueryMode => {
            let metadata = axes_metadata_snapshot_for_property(property, handle, index)
                .map_err(|err| limit_state_error("ylim", err))?;
            Ok(Value::String(
                if metadata.y_limits.is_some() {
                    "manual"
                } else {
                    "auto"
                }
                .into(),
            ))
        }
        LimitCommand::QueryMethod => query_limit_method("ylim", AxisLimitAxis::Y, &target),
        LimitCommand::SetMethod(method) => {
            set_limit_method("ylim", AxisLimitAxis::Y, method, &target)
        }
        LimitCommand::Freeze => {
            let metadata = axes_metadata_snapshot_for_property(property, handle, index)
                .map_err(|err| limit_state_error("ylim", err))?;
            let bounds =
                axis_display_bounds_snapshot_for_axes_for_property(property, handle, index)
                    .map_err(|err| limit_state_error("ylim", err))?;
            let limits = metadata
                .y_limits
                .or_else(|| bounds.map(|bounds| (bounds.2, bounds.3)))
                .unwrap_or((0.0, 1.0));
            set_axis_manual_limits_for_axes_for_property(
                property,
                handle,
                index,
                metadata.x_limits,
                Some(limits),
            )
            .map_err(|err| limit_state_error("ylim", err))?;
            Ok(limit_value(Some(limits)))
        }
        LimitCommand::Set(limits) => {
            let metadata = axes_metadata_snapshot_for_property(property, handle, index)
                .map_err(|err| limit_state_error("ylim", err))?;
            set_axis_manual_limits_for_axes_for_property(
                property,
                handle,
                index,
                metadata.x_limits,
                limits,
            )
            .map_err(|err| limit_state_error("ylim", err))?;
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
    fn ylim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = YLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = ylim()"));
        assert!(labels.contains(&"ylim([ymin ymax])"));
        assert!(labels.contains(&"mode = ylim('mode')"));
    }

    #[test]
    fn ylim_supports_auto_reset() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = ylim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![1.0, 5.0], vec![1, 2]).expect("y limits"),
        )])
        .unwrap();
        let manual = ylim_builtin(Vec::new()).expect("manual limits on empty axes");
        assert_eq!(
            runmat_value::Tensor::try_from(&manual)
                .expect("manual limits tensor")
                .materialize_f64(),
            vec![1.0, 5.0]
        );
        let _ = ylim_builtin(vec![Value::String("auto".into())]).unwrap();
        let queried = ylim_builtin(Vec::new()).unwrap();
        let tensor = runmat_value::Tensor::try_from(&queried).unwrap();
        assert!(tensor.materialize_f64()[0].is_nan() && tensor.materialize_f64()[1].is_nan());
    }
}
