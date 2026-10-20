use runmat_builtins::{XLIM_DESCRIPTOR, XLIM_ERROR_TOO_MANY_OUTPUTS};
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
    name = "xlim",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::xlim"
)]
pub fn xlim_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    xlim_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn xlim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    xlim_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn xlim_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &XLIM_DESCRIPTOR,
        &XLIM_ERROR_TOO_MANY_OUTPUTS,
        "xlim",
    )?;
    let call = prepare_limit_call(runtime, "xlim", args.view(), false)?;

    if matches!(
        call.command,
        LimitCommand::SetMethod(_) | LimitCommand::Freeze | LimitCommand::Set(_)
    ) {
        super::op_common::axes_output::validate_setter_output("xlim")?;
    }
    let target = call.target.resolve("xlim")?;
    let (handle, index) = target.coordinates();
    let property = target.authority();
    match call.command {
        LimitCommand::Query => {
            let limits = axis_displayed_limits_snapshot_for_axes_for_property(
                property,
                handle,
                index,
                AxisLimitAxis::X,
            )
            .map_err(|err| limit_state_error("xlim", err))?;
            Ok(limit_value(limits))
        }
        LimitCommand::QueryMode => {
            let metadata = axes_metadata_snapshot_for_property(property, handle, index)
                .map_err(|err| limit_state_error("xlim", err))?;
            Ok(Value::String(
                if metadata.x_limits.is_some() {
                    "manual"
                } else {
                    "auto"
                }
                .into(),
            ))
        }
        LimitCommand::QueryMethod => query_limit_method("xlim", AxisLimitAxis::X, &target),
        LimitCommand::SetMethod(method) => {
            set_limit_method("xlim", AxisLimitAxis::X, method, &target)
        }
        LimitCommand::Freeze => {
            let metadata = axes_metadata_snapshot_for_property(property, handle, index)
                .map_err(|err| limit_state_error("xlim", err))?;
            let bounds =
                axis_display_bounds_snapshot_for_axes_for_property(property, handle, index)
                    .map_err(|err| limit_state_error("xlim", err))?;
            let limits = metadata
                .x_limits
                .or_else(|| bounds.map(|bounds| (bounds.0, bounds.1)))
                .unwrap_or((0.0, 1.0));
            set_axis_manual_limits_for_axes_for_property(
                property,
                handle,
                index,
                Some(limits),
                metadata.y_limits,
            )
            .map_err(|err| limit_state_error("xlim", err))?;
            Ok(limit_value(Some(limits)))
        }
        LimitCommand::Set(limits) => {
            let metadata = axes_metadata_snapshot_for_property(property, handle, index)
                .map_err(|err| limit_state_error("xlim", err))?;
            set_axis_manual_limits_for_axes_for_property(
                property,
                handle,
                index,
                limits,
                metadata.y_limits,
            )
            .map_err(|err| limit_state_error("xlim", err))?;
            Ok(limit_value(limits))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::IntegerStorage;

    #[test]
    fn xlim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = XLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = xlim()"));
        assert!(labels.contains(&"xlim([xmin xmax])"));
        assert!(labels.contains(&"mode = xlim('mode')"));
    }

    #[test]
    fn xlim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let result = xlim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![0.0, 10.0], vec![1, 2]).expect("x limits"),
        )])
        .unwrap();
        assert!(matches!(result, Value::Tensor(_)));
        let queried = xlim_builtin(Vec::new()).unwrap();
        let tensor = runmat_value::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![0.0, 10.0]);
    }

    #[test]
    fn automatic_method_and_manual_freeze_share_displayed_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            Value::Tensor(
                runmat_value::Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).expect("line data"),
            ),
        ]))
        .expect("nonempty plot");

        xlim_builtin(vec![Value::String("tight".into())]).expect("tight method");
        let tight = runmat_value::Tensor::try_from(&xlim_builtin(Vec::new()).expect("tight query"))
            .expect("tight tensor")
            .materialize_f64();
        xlim_builtin(vec![Value::String("padded".into())]).expect("padded method");
        let padded =
            runmat_value::Tensor::try_from(&xlim_builtin(Vec::new()).expect("padded query"))
                .expect("padded tensor")
                .materialize_f64();
        assert!(padded[0] < tight[0] && padded[1] > tight[1]);
        assert!(matches!(
            xlim_builtin(vec![Value::String("method".into())]).expect("method"),
            Value::String(method) if method == "padded"
        ));
        xlim_builtin(vec![Value::String("manual".into())]).expect("freeze");
        assert!(matches!(
            xlim_builtin(vec![Value::String("mode".into())]).expect("mode"),
            Value::String(mode) if mode == "manual"
        ));
        let frozen = runmat_value::Tensor::try_from(&xlim_builtin(Vec::new()).expect("frozen"))
            .expect("frozen tensor")
            .materialize_f64();
        assert_eq!(frozen, padded);
    }

    #[test]
    fn xlim_rejects_integer_limits_that_collapse_in_graphics_coordinates() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let base = 1_u64 << 63;
        let error = xlim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new_integer(
                IntegerStorage::U64(vec![base, base + 1]),
                vec![1, 2],
            )
            .expect("wide limits"),
        )])
        .expect_err("graphics limits must remain distinct");
        assert!(error.message.contains("graphics coordinate domain"));
    }
}
