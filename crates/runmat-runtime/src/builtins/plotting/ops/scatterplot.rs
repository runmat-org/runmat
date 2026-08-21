//! MATLAB-compatible Communications Toolbox `scatterplot` builtin.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::state::{figure_handle_exists, select_figure, FigureHandle};
use crate::builtins::plotting::state::{set_axis_equal, set_axis_limits, set_grid_enabled};
use crate::builtins::plotting::style::{parse_line_style_args, LineStyleParseOptions};
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

const SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER_FIG: [BuiltinParamDescriptor; 5] = [
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
        name: "scatfig",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing figure handle used for the scatter plot.",
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
        label: "h = scatterplot(x, n, offset, marker, scatfig)",
        inputs: &SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER_FIG,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
];

const SCATTERPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatterplot:InvalidArgument"),
    when: "The input samples, decimation factor, offset, marker, or existing Figure handle is invalid.",
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

const SCATTERPLOT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "scatterplot-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "scatterplot accepts typed-integer sample data as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ScatterplotIntegerDataExtension"),
};
const SCATTERPLOT_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "scatterplot-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "scatterplot accepts typed-integer n and offset controls as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ScatterplotIntegerControlExtension"),
    };
pub const SCATTERPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SCATTERPLOT_INTEGER_DATA_EXTENSION,
    SCATTERPLOT_INTEGER_CONTROL_EXTENSION,
];
const SCATTERPLOT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double samples; typed integer values require exact binary64 representation before constellation geometry is computed.",
    }];
const SCATTERPLOT_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target specifies a double positive-integer decimation factor; RunMat mode validates typed integer storage exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "offset",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target specifies a double nonnegative-integer offset; RunMat mode validates it exactly and requires offset < n.",
    },
];
pub const SCATTERPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatterplot(integer_x, ...)",
        inputs: &SCATTERPLOT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The RunMat-only sample extension crosses into floating plotting coordinates only after exactness checks; the output is a double graphics handle.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatterplot(x, integer_n [, integer_offset], ...)",
        inputs: &SCATTERPLOT_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Decimation controls remain exact usize-bounded integers and are never routed through binary64.",
    },
];

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
    summary = "Plot complex samples as 2-D constellation points.",
    keywords = "scatterplot,constellation,communications,scatter,plotting,gpu",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::scatterplot::SCATTERPLOT_DESCRIPTOR),
    extensions(crate::builtins::plotting::scatterplot::SCATTERPLOT_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::scatterplot::SCATTERPLOT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::scatterplot"
)]
pub async fn scatterplot_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<f64> {
    crate::builtins::common::validation::reject_typed_complex_integer(&x, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &x,
        &SCATTERPLOT_INTEGER_DATA_EXTENSION,
        BUILTIN_NAME,
        "sample",
    )
    .await?;
    for control in rest.iter().take(2) {
        if crate::builtins::common::validation::value_has_native_integer_class(control) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SCATTERPLOT_INTEGER_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    let options = ScatterplotOptions::parse(rest)?;
    let (x_value, y_value, limits) = extract_scatter_values(x, options.n, options.offset).await?;
    let marker = options
        .marker
        .unwrap_or_else(|| Value::String(DEFAULT_MARKER.to_string()));

    if let Some(figure) = options.figure {
        select_figure(figure);
    }
    let handle = scatter_builtin(x_value, y_value, vec![marker])
        .await
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
    figure: Option<FigureHandle>,
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
        let marker = match rest.get(2) {
            Some(value) => Some(parse_marker(value)?),
            None => None,
        };
        let figure = match rest.get(3) {
            Some(value) => Some(parse_figure_handle(value)?),
            None => None,
        };
        Ok(Self {
            n,
            offset,
            marker,
            figure,
        })
    }
}

fn parse_marker(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::String(_) => {}
        Value::CharArray(chars) if chars.rows == 1 => {}
        Value::CharArray(_) => {
            return Err(scatterplot_error(
                "scatterplot: marker must be a character row vector or string scalar",
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }
        other => {
            return Err(scatterplot_error(
                format!("scatterplot: marker must be a LineSpec string, got {other:?}"),
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }
    }

    let opts = LineStyleParseOptions {
        builtin_name: BUILTIN_NAME,
        forbid_leading_numeric: true,
        forbid_interleaved_numeric: true,
        accepts_handle_visibility: false,
    };
    parse_line_style_args(std::slice::from_ref(value), &opts).map_err(|err| {
        scatterplot_error(
            format!("scatterplot: invalid marker LineSpec: {}", err.message()),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok(value.clone())
}

fn parse_figure_handle(value: &Value) -> BuiltinResult<FigureHandle> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        return Err(scatterplot_error(
            "scatterplot: scatfig must be a double Figure handle",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let scalar = parse_numeric_scalar(value, "scatfig")?;
    if !scalar.is_finite() || scalar <= 0.0 || scalar.fract() != 0.0 || scalar > f64::from(u32::MAX)
    {
        return Err(scatterplot_error(
            "scatterplot: scatfig must be a valid Figure handle",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let figure = FigureHandle::from(scalar as u32);
    if !figure_handle_exists(figure) {
        return Err(scatterplot_error(
            "scatterplot: scatfig must identify an existing Figure",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(figure)
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
            let (real, imag, limits) = extract_host_points(gathered, n, offset)?;
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_gpu_value(handle).await?;
            let (real, imag, limits) = extract_host_points(gathered, n, offset)?;
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
        other => {
            let (real, imag, limits) = extract_host_points(other, n, offset)?;
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
    }
}

async fn gpu_real_imag_handles(
    handle: &GpuTensorHandle,
) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)?;
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

type HostPointTensors = (Tensor, Tensor, Option<(f64, f64)>);

fn extract_host_points(value: Value, n: usize, offset: usize) -> BuiltinResult<HostPointTensors> {
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
    let limits = symmetric_limits(&real, &imag);
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
    Ok((x, y, limits))
}

fn complex_samples(value: Value) -> BuiltinResult<Vec<(f64, f64)>> {
    match value {
        Value::Complex(re, im) => Ok(vec![(re, im)]),
        Value::ComplexTensor(tensor) => Ok(tensor.materialize_f64()),
        Value::Num(v) => Ok(vec![(v, 0.0)]),
        Value::Int(v) => Ok(vec![(v.to_f64(), 0.0)]),
        Value::Bool(v) => Ok(vec![(if v { 1.0 } else { 0.0 }, 0.0)]),
        Value::Tensor(tensor) => Ok(tensor_utils::tensor_values_f64(&tensor)
            .into_iter()
            .map(|v| (v, 0.0))
            .collect()),
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
    if let Some(integer) = tensor_utils::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            scatterplot_error(
                format!("scatterplot: {name} is too large"),
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            )
        });
    }
    let scalar = parse_numeric_scalar(value, name)?;
    if !scalar.is_finite() || scalar < 0.0 || scalar.fract() != 0.0 {
        return Err(scatterplot_error(
            format!("scatterplot: {name} must be a nonnegative integer scalar"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    if scalar > usize::MAX as f64 || (usize::BITS == 64 && scalar == usize::MAX as f64) {
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
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
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
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use futures::executor::block_on;
    use runmat_plot::plots::{scatter::MarkerStyle, PlotElement};
    use runmat_value::ComplexTensor;
    use runmat_value::{IntegerStorage, NumericStorage};

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

    fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("integer tensor")
    }

    fn all_integer_scalar_storages(value: u8) -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![value as i8]),
            IntegerStorage::I16(vec![value as i16]),
            IntegerStorage::I32(vec![value as i32]),
            IntegerStorage::I64(vec![value as i64]),
            IntegerStorage::U8(vec![value]),
            IntegerStorage::U16(vec![value as u16]),
            IntegerStorage::U32(vec![value as u32]),
            IntegerStorage::U64(vec![value as u64]),
        ]
    }

    #[test]
    fn scatterplot_decimates_from_zero_based_offset() {
        let data = complex_tensor(&[(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]);
        let (x, y, limits) =
            extract_host_points(Value::ComplexTensor(data), 2, 1).expect("decimated points");
        assert_eq!(x.as_f64_slice(), Some(&[2.0, 4.0][..]));
        assert_eq!(y.as_f64_slice(), Some(&[20.0, 40.0][..]));
        assert_eq!(limits, Some((-42.0, 42.0)));
    }

    #[test]
    fn scatterplot_samples_and_scalars_read_typed_integer_storage() {
        let samples = poisoned_integer_tensor(IntegerStorage::I16(vec![1, -2, 3]), vec![3, 1]);
        let (x, y, _) = extract_host_points(Value::Tensor(samples), 1, 0).expect("integer samples");
        assert_eq!(x.as_f64_slice(), Some(&[1.0, -2.0, 3.0][..]));
        assert_eq!(y.as_f64_slice(), Some(&[0.0, 0.0, 0.0][..]));

        let decimation = poisoned_integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]);
        assert_eq!(
            parse_nonnegative_integer(&Value::Tensor(decimation), "n").expect("decimation"),
            2
        );
    }

    #[test]
    fn scatterplot_materializes_native_single_at_the_plotting_boundary() {
        let samples =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, -2.5]), vec![2, 1])
                .expect("single samples");
        let (x, y, limits) =
            extract_host_points(Value::Tensor(samples), 1, 0).expect("single samples");
        assert_eq!(x.as_f64_slice(), Some(&[1.25, -2.5][..]));
        assert_eq!(y.as_f64_slice(), Some(&[0.0, 0.0][..]));
        assert_eq!(limits, Some((-2.625, 2.625)));
    }

    #[test]
    fn scatterplot_count_parser_reads_all_integer_storages_without_mirrors() {
        for storage in all_integer_scalar_storages(2) {
            let value = Value::Tensor(poisoned_integer_tensor(storage, vec![1, 1]));
            assert_eq!(parse_nonnegative_integer(&value, "n").unwrap(), 2);
        }
    }

    #[test]
    fn scatterplot_figure_parser_rejects_all_integer_storages_without_mirrors() {
        for storage in all_integer_scalar_storages(1) {
            let value = Value::Tensor(poisoned_integer_tensor(storage, vec![1, 1]));
            assert!(parse_figure_handle(&value).is_err());
        }
    }

    #[test]
    fn scatterplot_rejects_offset_equal_to_decimation() {
        let err = ScatterplotOptions::parse(vec![Value::Num(2.0), Value::Num(2.0)]).unwrap_err();
        assert!(err.to_string().contains("offset must be less than n"));
    }

    #[test]
    fn scatterplot_rejects_invalid_marker_at_parse_time() {
        let err =
            ScatterplotOptions::parse(vec![Value::Num(1.0), Value::Num(0.0), Value::Num(7.0)])
                .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:scatterplot:InvalidArgument"));
        assert!(!err.to_string().contains("PlotFailed"));
    }

    #[test]
    fn scatterplot_rejects_unknown_figure_at_parse_time() {
        let err = ScatterplotOptions::parse(vec![
            Value::Num(1.0),
            Value::Num(0.0),
            Value::String("x".into()),
            Value::Num(f64::from(u32::MAX)),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:scatterplot:InvalidArgument"));
        assert!(!err.to_string().contains("PlotFailed"));
    }

    #[test]
    fn scatterplot_rejects_nonscalar_figure_at_parse_time() {
        let figure = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
        let err = ScatterplotOptions::parse(vec![
            Value::Num(1.0),
            Value::Num(0.0),
            Value::String("x".into()),
            Value::Tensor(figure),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:scatterplot:InvalidArgument"));
        assert!(!err.to_string().contains("PlotFailed"));
    }

    #[test]
    fn scatterplot_rejects_unrepresentable_integer_options_before_cast() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(parse_nonnegative_integer(&Value::Num(boundary), "n").is_err());
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
        let (x, y) = plot.host_xy_f64().unwrap().unwrap();
        assert_eq!(x, vec![1.0, 0.5, -1.0]);
        assert_eq!(y, vec![-1.0, 0.5, 1.0]);
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
    fn scatterplot_accepts_trailing_existing_figure_handle() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        let fig_handle = current_figure_handle();
        select_figure(fig_handle);
        let data = complex_tensor(&[(1.0, 2.0), (3.0, 4.0)]);
        let _ = run_scatterplot(
            Value::ComplexTensor(data),
            vec![
                Value::Num(1.0),
                Value::Num(0.0),
                Value::String("x".into()),
                Value::Num(f64::from(fig_handle.as_u32())),
            ],
        )
        .unwrap();
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[0]);
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
        assert!(labels.contains(&"h = scatterplot(x, n, offset, marker, scatfig)"));
    }

    #[test]
    fn scatterplot_is_registered_with_descriptor() {
        let builtin = runmat_builtins::builtin_function_by_name("scatterplot")
            .expect("scatterplot registered");
        assert_eq!(builtin.category, "communications/plotting");
        assert!(builtin.descriptor.is_some());
    }
}
