use runmat_builtins::{ZLIM_DESCRIPTOR, ZLIM_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::figure::AxisLimitAxis;
use runmat_value::Value;

use super::op_common::limits::{
    limit_state_error, limit_value, prepare_limit_call, query_limit_method, set_limit_method,
    LimitCommand,
};
use super::state::{
    axes_metadata_snapshot_for_property, axis_displayed_limits_snapshot_for_axes_for_property,
    axis_z_limits_snapshot_for_axes_for_property,
    set_axis_z_limits_preserving_selection_for_property,
};

#[runtime_builtin(
    name = "zlim",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::zlim"
)]
pub fn zlim_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    zlim_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn zlim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    zlim_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn zlim_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &ZLIM_DESCRIPTOR,
        &ZLIM_ERROR_TOO_MANY_OUTPUTS,
        "zlim",
    )?;
    let call = prepare_limit_call(runtime, "zlim", args.view(), false)?;

    if matches!(
        call.command,
        LimitCommand::SetMethod(_) | LimitCommand::Freeze | LimitCommand::Set(_)
    ) {
        super::op_common::axes_output::validate_setter_output("zlim")?;
    }
    let target = call.target.resolve("zlim")?;
    let (handle, index) = target.coordinates();
    let property = target.authority();
    match call.command {
        LimitCommand::Query => {
            let limits = axis_displayed_limits_snapshot_for_axes_for_property(
                property,
                handle,
                index,
                AxisLimitAxis::Z,
            )
            .map_err(|err| limit_state_error("zlim", err))?;
            Ok(limit_value(limits.or(Some((0.0, 1.0)))))
        }
        LimitCommand::QueryMode => {
            let metadata = axes_metadata_snapshot_for_property(property, handle, index)
                .map_err(|err| limit_state_error("zlim", err))?;
            Ok(Value::String(
                if metadata.z_limits.is_some() {
                    "manual"
                } else {
                    "auto"
                }
                .into(),
            ))
        }
        LimitCommand::QueryMethod => query_limit_method("zlim", AxisLimitAxis::Z, &target),
        LimitCommand::SetMethod(method) => {
            set_limit_method("zlim", AxisLimitAxis::Z, method, &target)
        }
        LimitCommand::Freeze => {
            let limits = axis_z_limits_snapshot_for_axes_for_property(property, handle, index)
                .map_err(|err| limit_state_error("zlim", err))?
                .unwrap_or((0.0, 1.0));
            set_axis_z_limits_preserving_selection_for_property(
                property,
                handle,
                index,
                Some(limits),
            )
            .map_err(|err| limit_state_error("zlim", err))?;
            Ok(limit_value(Some(limits)))
        }
        LimitCommand::Set(limits) => {
            set_axis_z_limits_preserving_selection_for_property(property, handle, index, limits)
                .map_err(|err| limit_state_error("zlim", err))?;
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
    fn zlim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ZLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = zlim()"));
        assert!(labels.contains(&"zlim([zmin zmax])"));
        assert!(labels.contains(&"mode = zlim('mode')"));
    }

    #[test]
    fn zlim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = zlim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![-1.0, 1.0], vec![1, 2]).expect("z limits"),
        )])
        .unwrap();
        let queried = zlim_builtin(Vec::new()).unwrap();
        let tensor = runmat_value::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![-1.0, 1.0]);
    }
}
