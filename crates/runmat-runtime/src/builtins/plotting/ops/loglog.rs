use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{LOGLOG_ERROR_LOG_AXIS_FAILED, LOGLOG_ERROR_PLOT_FAILED};
use runmat_macros::runtime_builtin;

use super::plot::plot_builtin_with_runtime;
use super::state::{current_axes_state, set_log_modes_for_axes, FigureError};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "loglog";

fn loglog_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn loglog_map_plot_error(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        LOGLOG_ERROR_PLOT_FAILED.message,
        err.message()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = LOGLOG_ERROR_PLOT_FAILED.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

fn loglog_map_axes_error(err: FigureError) -> RuntimeError {
    loglog_error_with_message(
        format!("{}: {}", LOGLOG_ERROR_LOG_AXIS_FAILED.message, err),
        &LOGLOG_ERROR_LOG_AXIS_FAILED,
    )
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::loglog")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "loglog",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "loglog is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::loglog")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "loglog",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "loglog performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "loglog",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::loglog"
)]
pub async fn loglog_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::LOGLOG_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "loglog",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "loglog")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(loglog_impl(runtime.as_ref(), args, &work).await)
}

async fn loglog_impl(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let result = plot_builtin_with_runtime(runtime, args)
        .await
        .map_err(loglog_map_plot_error)?;
    work.check()?;
    let axes = current_axes_state();
    set_log_modes_for_axes(axes.handle, axes.active_index, true, true)
        .map_err(loglog_map_axes_error)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use runmat_builtins::LOGLOG_DESCRIPTOR;
    async fn loglog_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::loglog_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use crate::builtins::plotting::{configure_subplot, current_figure_handle};
    use runmat_value::{Tensor, Value};

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("loglog test vector")
    }

    #[test]
    fn loglog_sets_both_axes_log_on_active_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(loglog_builtin(vec![Value::Tensor(tensor_from(&[
            1.0, 10.0, 100.0,
        ]))]));
        let fig = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert!(meta.x_log);
        assert!(meta.y_log);
    }

    #[test]
    fn loglog_accepts_leading_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();

        let _ = futures::executor::block_on(loglog_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[1.0, 10.0])),
        ]));
        let fig = clone_figure(fig_handle).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert!(meta.x_log);
        assert!(meta.y_log);
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn loglog_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LOGLOG_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = loglog(Y)"));
        assert!(labels.contains(&"h = loglog(X, Y)"));
        assert!(labels.contains(&"h = loglog(args...)"));
    }
}
