use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{SEMILOGX_ERROR_LOG_AXIS_FAILED, SEMILOGX_ERROR_PLOT_FAILED};
use runmat_macros::runtime_builtin;

use super::plot::plot_builtin_with_runtime;
use super::state::{current_axes_state, set_log_modes_for_axes, FigureError};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "semilogx";

fn semilogx_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn semilogx_map_plot_error(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        SEMILOGX_ERROR_PLOT_FAILED.message,
        err.message()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = SEMILOGX_ERROR_PLOT_FAILED.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

fn semilogx_map_axes_error(err: FigureError) -> RuntimeError {
    semilogx_error_with_message(
        format!("{}: {}", SEMILOGX_ERROR_LOG_AXIS_FAILED.message, err),
        &SEMILOGX_ERROR_LOG_AXIS_FAILED,
    )
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::semilogx")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "semilogx",
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
    notes: "semilogx is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::semilogx")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "semilogx",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "semilogx performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "semilogx",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::semilogx"
)]
pub async fn semilogx_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SEMILOGX_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "semilogx",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "semilogx")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(semilogx_impl(runtime.as_ref(), args, &work).await)
}

async fn semilogx_impl(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let result = plot_builtin_with_runtime(runtime, args)
        .await
        .map_err(semilogx_map_plot_error)?;
    work.check()?;
    let axes = current_axes_state();
    set_log_modes_for_axes(axes.handle, axes.active_index, true, false)
        .map_err(semilogx_map_axes_error)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use runmat_builtins::SEMILOGX_DESCRIPTOR;
    async fn semilogx_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::semilogx_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_value::{Tensor, Value};

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("semilogx test vector")
    }

    #[test]
    fn semilogx_sets_x_log_on_active_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(semilogx_builtin(vec![Value::Tensor(tensor_from(&[
            1.0, 10.0, 100.0,
        ]))]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert!(meta.x_log);
        assert!(!meta.y_log);
    }

    #[test]
    fn semilogx_is_subplot_local() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();

        let _ = futures::executor::block_on(semilogx_builtin(vec![Value::Tensor(tensor_from(&[
            1.0, 10.0,
        ]))]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(!fig.axes_metadata(0).unwrap().x_log);
        assert!(fig.axes_metadata(1).unwrap().x_log);
    }

    #[test]
    fn semilogx_accepts_leading_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();

        let _ = futures::executor::block_on(semilogx_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[1.0, 10.0])),
        ]));
        let fig = clone_figure(fig_handle).unwrap();
        assert!(fig.axes_metadata(1).unwrap().x_log);
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn semilogx_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SEMILOGX_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = semilogx(Y)"));
        assert!(labels.contains(&"h = semilogx(X, Y)"));
        assert!(labels.contains(&"h = semilogx(args...)"));
    }
}
