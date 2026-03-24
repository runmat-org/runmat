use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::plot::plot_builtin;
use super::state::{current_axes_state, set_log_modes_for_axes};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::string_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::semilogx")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "semilogx",
    op_kind: GpuOpKind::Custom("plot-render"),
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
    category = "plotting",
    summary = "Create a plot with logarithmic X axis.",
    keywords = "semilogx,plotting,log",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::semilogx"
)]
pub async fn semilogx_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let result = plot_builtin(args).await;
    let axes = current_axes_state();
    set_log_modes_for_axes(axes.handle, axes.active_index, true, false).map_err(|err| {
        crate::builtins::plotting::plotting_error_with_source(
            "semilogx",
            format!("semilogx: {err}"),
            err,
        )
    })?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_builtins::{NumericDType, Tensor};

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: NumericDType::F64,
        }
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
}
