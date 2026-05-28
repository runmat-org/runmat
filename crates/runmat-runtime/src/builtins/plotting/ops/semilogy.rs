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

const BUILTIN_NAME: &str = "semilogy";

const SEMILOGY_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line handle.",
}];
const SEMILOGY_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y data vector/matrix.",
}];
const SEMILOGY_INPUTS_XY: [BuiltinParamDescriptor; 2] = [
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
const SEMILOGY_INPUTS_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Plot-style inputs: optional axes handle, style tokens, and Name/Value pairs.",
}];
const SEMILOGY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "h = semilogy(Y)",
        inputs: &SEMILOGY_INPUTS_Y,
        outputs: &SEMILOGY_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = semilogy(X, Y)",
        inputs: &SEMILOGY_INPUTS_XY,
        outputs: &SEMILOGY_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = semilogy(args...)",
        inputs: &SEMILOGY_INPUTS_ARGS,
        outputs: &SEMILOGY_OUTPUT_HANDLE,
    },
];
const SEMILOGY_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SEMILOGY.PLOT_FAILED",
    identifier: Some("RunMat:semilogy:PlotFailed"),
    when: "Underlying plot rendering/parsing fails while building the semilog plot.",
    message: "semilogy: plot operation failed",
};
const SEMILOGY_ERROR_LOG_AXIS_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SEMILOGY.LOG_AXIS_FAILED",
    identifier: Some("RunMat:semilogy:LogAxisFailed"),
    when: "Applying logarithmic Y-axis mode fails.",
    message: "semilogy: failed to apply logarithmic axis mode",
};
const SEMILOGY_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SEMILOGY_ERROR_PLOT_FAILED, SEMILOGY_ERROR_LOG_AXIS_FAILED];
pub const SEMILOGY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SEMILOGY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SEMILOGY_ERRORS,
};

fn semilogy_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn semilogy_map_plot_error(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        SEMILOGY_ERROR_PLOT_FAILED.message,
        err.message()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = SEMILOGY_ERROR_PLOT_FAILED.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

fn semilogy_map_axes_error(err: FigureError) -> RuntimeError {
    semilogy_error_with_message(
        format!("{}: {}", SEMILOGY_ERROR_LOG_AXIS_FAILED.message, err),
        &SEMILOGY_ERROR_LOG_AXIS_FAILED,
    )
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::semilogy")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "semilogy",
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
    notes: "semilogy is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::semilogy")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "semilogy",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "semilogy performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "semilogy",
    category = "plotting",
    summary = "Create a plot with logarithmic Y axis.",
    keywords = "semilogy,plotting,log",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::semilogy::SEMILOGY_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::semilogy"
)]
pub async fn semilogy_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let result = plot_builtin(args).await.map_err(semilogy_map_plot_error)?;
    let axes = current_axes_state();
    set_log_modes_for_axes(axes.handle, axes.active_index, false, true)
        .map_err(semilogy_map_axes_error)?;
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
    fn semilogy_sets_y_log_on_active_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(semilogy_builtin(vec![Value::Tensor(tensor_from(&[
            1.0, 10.0, 100.0,
        ]))]));
        let fig = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert!(!meta.x_log);
        assert!(meta.y_log);
    }

    #[test]
    fn semilogy_accepts_leading_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();

        let _ = futures::executor::block_on(semilogy_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[1.0, 10.0])),
        ]));
        let fig = clone_figure(fig_handle).unwrap();
        assert!(fig.axes_metadata(1).unwrap().y_log);
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn semilogy_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SEMILOGY_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = semilogy(Y)"));
        assert!(labels.contains(&"h = semilogy(X, Y)"));
        assert!(labels.contains(&"h = semilogy(args...)"));
    }
}
