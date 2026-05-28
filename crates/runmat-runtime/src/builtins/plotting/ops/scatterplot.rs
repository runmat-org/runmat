//! MATLAB-compatible Communications Toolbox `scatterplot` builtin.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::ComplexTensor;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::state::{set_axis_equal, set_axis_limits, set_grid_enabled};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::gpu_helpers::axis_bounds;
use super::scatter::scatter_builtin;

const BUILTIN_NAME: &str = "scatterplot";
const DEFAULT_MARKER: &str = "b.";

const SCATTERPLOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure/graphics handle returned by the underlying scatter plot.",
}];

const SCATTERPLOT_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex-baseband samples. Real inputs are plotted with zero imaginary part.",
}];

const SCATTERPLOT_INPUTS_X_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor; every n-th sample is plotted.",
    },
];

const SCATTERPLOT_INPUTS_X_N_OFFSET: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor; every n-th sample is plotted.",
    },
    BuiltinParamDescriptor {
        name: "offset",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based sample offset before decimation.",
    },
];

const SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor.",
    },
    BuiltinParamDescriptor {
        name: "offset",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based sample offset before decimation.",
    },
    BuiltinParamDescriptor {
        name: "marker",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker LineSpec forwarded to scatter.",
    },
];

const SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER_AX: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor.",
    },
    BuiltinParamDescriptor {
        name: "offset",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based sample offset before decimation.",
    },
    BuiltinParamDescriptor {
        name: "marker",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker LineSpec forwarded to scatter.",
    },
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
];

const SCATTERPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x)",
        inputs: &SCATTERPLOT_INPUTS_X,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n)",
        inputs: &SCATTERPLOT_INPUTS_X_N,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n, offset)",
        inputs: &SCATTERPLOT_INPUTS_X_N_OFFSET,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n, offset, marker)",
        inputs: &SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n, offset, marker, ax)",
        inputs: &SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER_AX,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
];

const SCATTERPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatterplot:InvalidArgument"),
    when: "The input samples, decimation factor, offset, marker, or axes handle is invalid.",
    message: "scatterplot: invalid argument",
};

const SCATTERPLOT_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERPLOT.PLOT_FAILED",
    identifier: Some("RunMat:scatterplot:PlotFailed"),
    when: "The underlying scatter plot cannot be rendered.",
    message: "scatterplot: plot operation failed",
};

const SCATTERPLOT_ERRORS: [BuiltinErrorDescriptor; 2] = [
    SCATTERPLOT_ERROR_INVALID_ARGUMENT,
    SCATTERPLOT_ERROR_PLOT_FAILED,
];

pub const SCATTERPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SCATTERPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SCATTERPLOT_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::scatterplot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "scatterplot",
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
    notes: "scatterplot forwards GPU-resident real/imag buffers to scatter for zero-copy plotting when no decimation is requested; decimated or unsupported inputs gather once.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::scatterplot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "scatterplot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "scatterplot is a rendering sink and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "scatterplot",
    category = "communications/plotting",
    summary = "Plot complex-baseband constellation samples.",
    keywords = "scatterplot,constellation,communications,scatter,plotting,gpu",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::scatterplot::SCATTERPLOT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::scatterplot"
)]
pub async fn scatterplot_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<f64> {
    let options = ScatterplotOptions::parse(rest)?;
    let (x_value, y_value, limits) = extract_scatter_values(x, options.n, options.offset).await?;
    let marker = options
        .marker
        .unwrap_or_else(|| Value::String(DEFAULT_MARKER.to_string()));

    let handle = if let Some(ax) = options.axes {
        scatter_builtin(Value::Num(ax), x_value, vec![y_value, marker]).await
    } else {
        scatter_builtin(x_value, y_value, vec![marker]).await
    }
    .map_err(scatterplot_map_plot_error)?;
    set_axis_equal(true);
    set_grid_enabled(true);
    if let Some(limit) = limits {
        set_axis_limits(Some(limit), Some(limit));
    }
    Ok(handle)
}

#[derive(Clone, Debug)]
struct ScatterplotOptions {
    n: usize,
    offset: usize,
    marker: Option<Value>,
    axes: Option<f64>,
}

impl ScatterplotOptions {
    fn parse(rest: Vec<Value>) -> BuiltinResult<Self> {
        if rest.len() > 4 {
            return Err(scatterplot_error(
                SCATTERPLOT_ERROR_INVALID_ARGUMENT.message,
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }

        let n = match rest.first() {
            Some(value) => parse_nonnegative_integer(value, "n").and_then(|n| {
                if n == 0 {
                    Err(scatterplot_error(
                        "scatterplot: n must be a positive integer",
                        &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
                    ))
                } else {
                    Ok(n)
                }
            })?,
            None => 1,
        };
        let offset = match rest.get(1) {
            Some(value) => parse_nonnegative_integer(value, "offset")?,
            None => 0,
        };
        if offset >= n {
            return Err(scatterplot_error(
                "scatterplot: offset must be less than n",
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }
        let marker = rest.get(2).cloned();
        let axes = match rest.get(3) {
            Some(value) => Some(parse_numeric_scalar(value, "ax")?),
            None => None,
        };
        Ok(Self {
            n,
            offset,
            marker,
            axes,
        })
    }
}

async fn extract_scatter_values(
    value: Value,
    n: usize,
    offset: usize,
) -> BuiltinResult<(Value, Value, Option<(f64, f64)>)> {
    match value {
        Value::GpuTensor(handle) if n == 1 && offset == 0 => {
            if let Some((real, imag)) = gpu_real_imag_handles(&handle).await {
                let limits = symmetric_limits_from_gpu_bounds(&real, &imag);
                return Ok((Value::GpuTensor(real), Value::GpuTensor(imag), limits));
            }
            let gathered = gather_gpu_value(handle).await?;
            let (real, imag) = extract_host_points(gathered, n, offset)?;
            let limits = symmetric_limits(&real.data, &imag.data);
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_gpu_value(handle).await?;
            let (real, imag) = extract_host_points(gathered, n, offset)?;
            let limits = symmetric_limits(&real.data, &imag.data);
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
        other => {
            let (real, imag) = extract_host_points(other, n, offset)?;
            let limits = symmetric_limits(&real.data, &imag.data);
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
    }
}

async fn gpu_real_imag_handles(
    handle: &GpuTensorHandle,
) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
    let provider = runmat_accelerate_api::provider_for_handle(handle)?;
    let real = provider.unary_real(handle).await.ok()?;
    let imag = provider.unary_imag(handle).await.ok()?;
    Some((real, imag))
}

async fn gather_gpu_value(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let value = Value::GpuTensor(handle);
    crate::gather_if_needed_async(&value)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
}

fn extract_host_points(value: Value, n: usize, offset: usize) -> BuiltinResult<(Tensor, Tensor)> {
    let samples = complex_samples(value)?;
    if samples.is_empty() {
        return Err(scatterplot_error(
            "scatterplot: input samples cannot be empty",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let selected: Vec<(f64, f64)> = samples.into_iter().skip(offset).step_by(n).collect();
    if selected.is_empty() {
        return Err(scatterplot_error(
            "scatterplot: decimation selects no samples",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut real = Vec::with_capacity(selected.len());
    let mut imag = Vec::with_capacity(selected.len());
    for (re, im) in selected {
        real.push(re);
        imag.push(im);
    }
    let shape = vec![real.len(), 1];
    let x = Tensor::new(real, shape.clone()).map_err(|err| {
        scatterplot_error(
            format!("scatterplot: {err}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )
    })?;
    let y = Tensor::new(imag, shape).map_err(|err| {
        scatterplot_error(
            format!("scatterplot: {err}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok((x, y))
}

fn complex_samples(value: Value) -> BuiltinResult<Vec<(f64, f64)>> {
    match value {
        Value::Complex(re, im) => Ok(vec![(re, im)]),
        Value::ComplexTensor(tensor) => Ok(tensor.data),
        Value::Num(v) => Ok(vec![(v, 0.0)]),
        Value::Int(v) => Ok(vec![(v.to_f64(), 0.0)]),
        Value::Bool(v) => Ok(vec![(if v { 1.0 } else { 0.0 }, 0.0)]),
        Value::Tensor(tensor) => Ok(tensor.data.into_iter().map(|v| (v, 0.0)).collect()),
        Value::LogicalArray(logical) => Ok(logical
            .data
            .into_iter()
            .map(|v| (if v != 0 { 1.0 } else { 0.0 }, 0.0))
            .collect()),
        other => Err(scatterplot_error(
            format!("scatterplot: expected numeric or complex samples, got {other:?}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn parse_nonnegative_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let scalar = parse_numeric_scalar(value, name)?;
    if !scalar.is_finite() || scalar < 0.0 || scalar.fract() != 0.0 {
        return Err(scatterplot_error(
            format!("scatterplot: {name} must be a nonnegative integer scalar"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    if scalar > (usize::MAX as f64) {
        return Err(scatterplot_error(
            format!("scatterplot: {name} is too large"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(scalar as usize)
}

fn parse_numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(v) => Ok(v.to_f64()),
        Value::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        other => Err(scatterplot_error(
            format!("scatterplot: {name} must be a numeric scalar, got {other:?}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn symmetric_limits(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let max_abs = x
        .iter()
        .chain(y.iter())
        .copied()
        .filter(|v| v.is_finite())
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if max_abs == 0.0 {
        Some((-1.0, 1.0))
    } else if max_abs.is_finite() {
        let padded = max_abs * 1.05;
        Some((-padded, padded))
    } else {
        None
    }
}

fn symmetric_limits_from_gpu_bounds(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
) -> Option<(f64, f64)> {
    let (xmin, xmax) = axis_bounds(x, BUILTIN_NAME).ok()?;
    let (ymin, ymax) = axis_bounds(y, BUILTIN_NAME).ok()?;
    symmetric_limits(&[xmin as f64, xmax as f64], &[ymin as f64, ymax as f64])
}

fn scatterplot_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn scatterplot_map_plot_error(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        SCATTERPLOT_ERROR_PLOT_FAILED.message,
        err.message()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = SCATTERPLOT_ERROR_PLOT_FAILED.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use futures::executor::block_on;
    use runmat_plot::plots::{scatter::MarkerStyle, PlotElement};

    fn setup_plot_tests() {
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
    }

    fn run_scatterplot(x: Value, rest: Vec<Value>) -> BuiltinResult<f64> {
        block_on(super::scatterplot_builtin(x, rest))
    }

    fn complex_tensor(data: &[(f64, f64)]) -> ComplexTensor {
        ComplexTensor::new(data.to_vec(), vec![data.len(), 1]).expect("complex tensor")
    }

    #[test]
    fn scatterplot_decimates_from_zero_based_offset() {
        let data = complex_tensor(&[(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]);
        let (x, y) =
            extract_host_points(Value::ComplexTensor(data), 2, 1).expect("decimated points");
        assert_eq!(x.data, vec![2.0, 4.0]);
        assert_eq!(y.data, vec![20.0, 40.0]);
    }

    #[test]
    fn scatterplot_rejects_offset_equal_to_decimation() {
        let err = ScatterplotOptions::parse(vec![Value::Num(2.0), Value::Num(2.0)]).unwrap_err();
        assert!(err.to_string().contains("offset must be less than n"));
    }

    #[test]
    fn scatterplot_smoke_renders_complex_samples() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        let data = complex_tensor(&[(1.0, -1.0), (0.5, 0.5), (-1.0, 1.0)]);
        let _ = run_scatterplot(Value::ComplexTensor(data), Vec::new()).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Scatter(plot) = fig.plots().next().unwrap() else {
            panic!("expected scatter plot")
        };
        assert_eq!(plot.x_data, vec![1.0, 0.5, -1.0]);
        assert_eq!(plot.y_data, vec![-1.0, 0.5, 1.0]);
        assert!(fig.axis_equal);
        assert!(fig.grid_enabled);
    }

    #[test]
    fn scatterplot_forwards_marker_to_scatter() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        let data = complex_tensor(&[(1.0, 1.0), (2.0, 2.0)]);
        let _ = run_scatterplot(
            Value::ComplexTensor(data),
            vec![Value::Num(1.0), Value::Num(0.0), Value::String("s".into())],
        )
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Scatter(plot) = fig.plots().next().unwrap() else {
            panic!("expected scatter plot")
        };
        assert_eq!(plot.marker_style, MarkerStyle::Square);
    }

    #[test]
    fn scatterplot_accepts_trailing_axes_handle() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        configure_subplot(1, 2, 0).unwrap();
        let data = complex_tensor(&[(1.0, 2.0), (3.0, 4.0)]);
        let _ = run_scatterplot(
            Value::ComplexTensor(data),
            vec![
                Value::Num(1.0),
                Value::Num(0.0),
                Value::String("x".into()),
                Value::Num(ax),
            ],
        )
        .unwrap();
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn scatterplot_descriptor_lists_matlab_call_forms() {
        let labels: Vec<&str> = SCATTERPLOT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = scatterplot(x)"));
        assert!(labels.contains(&"h = scatterplot(x, n)"));
        assert!(labels.contains(&"h = scatterplot(x, n, offset)"));
        assert!(labels.contains(&"h = scatterplot(x, n, offset, marker)"));
        assert!(labels.contains(&"h = scatterplot(x, n, offset, marker, ax)"));
    }

    #[test]
    fn scatterplot_is_registered_with_descriptor() {
        let builtin = runmat_builtins::builtin_function_by_name("scatterplot")
            .expect("scatterplot registered");
        assert_eq!(builtin.category, "communications/plotting");
        assert!(builtin.descriptor.is_some());
    }
}
