use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::plot::plot_builtin;
use super::state::{current_axes_state, set_log_modes_for_axes, FigureError};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "loglog";

const LOGLOG_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line handle.",
}];
const LOGLOG_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y data vector/matrix.",
}];
const LOGLOG_INPUTS_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data vector/matrix.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector/matrix.",
    },
];
const LOGLOG_INPUTS_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Plot-style inputs: optional axes handle, style tokens, and Name/Value pairs.",
}];
const LOGLOG_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "h = loglog(Y)",
        inputs: &LOGLOG_INPUTS_Y,
        outputs: &LOGLOG_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = loglog(X, Y)",
        inputs: &LOGLOG_INPUTS_XY,
        outputs: &LOGLOG_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = loglog(args...)",
        inputs: &LOGLOG_INPUTS_ARGS,
        outputs: &LOGLOG_OUTPUT_HANDLE,
    },
];
const LOGLOG_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGLOG.PLOT_FAILED",
    identifier: Some("RunMat:loglog:PlotFailed"),
    when: "Underlying plot rendering/parsing fails while building the log-log plot.",
    message: "loglog: plot operation failed",
};
const LOGLOG_ERROR_LOG_AXIS_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGLOG.LOG_AXIS_FAILED",
    identifier: Some("RunMat:loglog:LogAxisFailed"),
    when: "Applying logarithmic axis modes fails.",
    message: "loglog: failed to apply logarithmic axis mode",
};
const LOGLOG_ERRORS: [BuiltinErrorDescriptor; 2] =
    [LOGLOG_ERROR_PLOT_FAILED, LOGLOG_ERROR_LOG_AXIS_FAILED];
pub const LOGLOG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOGLOG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOGLOG_ERRORS,
};

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
    category = "plotting",
    summary = "Create a plot with logarithmic X and Y axes.",
    keywords = "loglog,plotting,log",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::loglog::LOGLOG_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::loglog"
)]
pub async fn loglog_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let result = plot_builtin(args).await.map_err(loglog_map_plot_error)?;
    let axes = current_axes_state();
    set_log_modes_for_axes(axes.handle, axes.active_index, true, true)
        .map_err(loglog_map_axes_error)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use crate::builtins::plotting::{configure_subplot, current_figure_handle};
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
