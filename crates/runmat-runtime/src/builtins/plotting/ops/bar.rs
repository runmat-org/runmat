//! MATLAB-compatible `bar` and `barh` builtins.

use futures::executor::block_on;
use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::bar::{BarGpuInputs, BarGpuParams, BarLayoutMode, BarOrientation};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::bar::Orientation;
use runmat_plot::plots::BarChart;
use std::convert::TryFrom;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::gather_if_needed_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::common::numeric_vector;
use super::gpu_helpers::axis_bounds;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarLayout, BarStyle, BarStyleDefaults};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "bar";
const BARH_BUILTIN_NAME: &str = "barh";

const BAR_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the first rendered bar series.",
}];

const BAR_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bar heights as a vector or matrix.",
}];

const BAR_INPUTS_Y_STYLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "style",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Layout/style token such as 'stacked' or color shorthand.",
    },
];

const BAR_INPUTS_Y_WIDTH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.75"),
        description: "Fraction of available category space occupied by each bar.",
    },
];

const BAR_INPUTS_Y_WIDTH_COLOR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.75"),
        description: "Fraction of available category space occupied by each bar.",
    },
    BuiltinParamDescriptor {
        name: "color",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Color shorthand or color name for all bars.",
    },
];

const BAR_INPUTS_Y_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style arguments.",
    },
];

const BAR_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
];

const BAR_INPUTS_X_Y_STYLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "style",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Layout/style token such as 'stacked' or color shorthand.",
    },
];

const BAR_INPUTS_X_Y_WIDTH: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.75"),
        description: "Fraction of available category space occupied by each bar.",
    },
];

const BAR_INPUTS_X_Y_WIDTH_COLOR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.75"),
        description: "Fraction of available category space occupied by each bar.",
    },
    BuiltinParamDescriptor {
        name: "color",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Color shorthand or color name for all bars.",
    },
];

const BAR_INPUTS_X_Y_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style arguments.",
    },
];

const BAR_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "h = bar(Y)",
        inputs: &BAR_INPUTS_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, width)",
        inputs: &BAR_INPUTS_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, width, color)",
        inputs: &BAR_INPUTS_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, style)",
        inputs: &BAR_INPUTS_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y)",
        inputs: &BAR_INPUTS_X_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, width)",
        inputs: &BAR_INPUTS_X_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, width, color)",
        inputs: &BAR_INPUTS_X_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, style)",
        inputs: &BAR_INPUTS_X_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_X_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
];

const BARH_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "h = barh(Y)",
        inputs: &BAR_INPUTS_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, width)",
        inputs: &BAR_INPUTS_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, width, color)",
        inputs: &BAR_INPUTS_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, style)",
        inputs: &BAR_INPUTS_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y)",
        inputs: &BAR_INPUTS_X_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, width)",
        inputs: &BAR_INPUTS_X_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, width, color)",
        inputs: &BAR_INPUTS_X_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, style)",
        inputs: &BAR_INPUTS_X_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_X_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
];

const BAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:bar:InvalidArgument"),
    when: "Bar input arrays, style tokens, or name/value arguments are invalid.",
    message: "bar: invalid argument",
};

const BAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAR.INTERNAL",
    identifier: Some("RunMat:bar:Internal"),
    when: "Renderer/GPU conversion fails while building bar geometry.",
    message: "bar: internal operation failed",
};

const BAR_ERRORS: [BuiltinErrorDescriptor; 2] = [BAR_ERROR_INVALID_ARGUMENT, BAR_ERROR_INTERNAL];

const BARH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BARH.INVALID_ARGUMENT",
    identifier: Some("RunMat:barh:InvalidArgument"),
    when: "Horizontal bar input arrays, style tokens, or name/value arguments are invalid.",
    message: "barh: invalid argument",
};

const BARH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BARH.INTERNAL",
    identifier: Some("RunMat:barh:Internal"),
    when: "Renderer/GPU conversion fails while building horizontal bar geometry.",
    message: "barh: internal operation failed",
};

const BARH_ERRORS: [BuiltinErrorDescriptor; 2] = [BARH_ERROR_INVALID_ARGUMENT, BARH_ERROR_INTERNAL];

pub const BAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BAR_ERRORS,
};

pub const BARH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BARH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BARH_ERRORS,
};

fn bar_descriptor_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: Option<impl AsRef<str>>,
) -> RuntimeError {
    let message = match detail {
        Some(detail) => format!("{}: {}", error.message, detail.as_ref()),
        None => error.message.to_string(),
    };
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Copy)]
struct BarRenderConfig {
    builtin_name: &'static str,
    title: &'static str,
    x_label: &'static str,
    y_label: &'static str,
    orientation: Orientation,
    gpu_orientation: BarOrientation,
    invalid_error: &'static BuiltinErrorDescriptor,
    internal_error: &'static BuiltinErrorDescriptor,
}

impl BarRenderConfig {
    fn invalid(self, detail: impl AsRef<str>) -> RuntimeError {
        bar_descriptor_error(self.builtin_name, self.invalid_error, Some(detail))
    }

    fn internal(self, detail: impl AsRef<str>) -> RuntimeError {
        bar_descriptor_error(self.builtin_name, self.internal_error, Some(detail))
    }
}

const BAR_CONFIG: BarRenderConfig = BarRenderConfig {
    builtin_name: BUILTIN_NAME,
    title: "Bar Chart",
    x_label: "Category",
    y_label: "Value",
    orientation: Orientation::Vertical,
    gpu_orientation: BarOrientation::Vertical,
    invalid_error: &BAR_ERROR_INVALID_ARGUMENT,
    internal_error: &BAR_ERROR_INTERNAL,
};

const BARH_CONFIG: BarRenderConfig = BarRenderConfig {
    builtin_name: BARH_BUILTIN_NAME,
    title: "Horizontal Bar Chart",
    x_label: "Value",
    y_label: "Category",
    orientation: Orientation::Horizontal,
    gpu_orientation: BarOrientation::Horizontal,
    invalid_error: &BARH_ERROR_INVALID_ARGUMENT,
    internal_error: &BARH_ERROR_INTERNAL,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "bar",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink, but can consume gpuArray inputs zero-copy when a shared WGPU context exists.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Bar charts terminate fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const BARH_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "barh",
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
    notes: "Horizontal bar charts terminate fusion graphs; gpuArray inputs may remain on device through the shared bar vertex-packing path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "bar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "bar performs I/O and terminates fusion graphs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const BARH_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "barh",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "barh performs I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "bar",
    category = "plotting",
    summary = "Create bar charts.",
    keywords = "bar,barchart,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::bar::BAR_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::bar"
)]
pub async fn bar_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    bar_impl(args, BAR_CONFIG).await
}

#[runtime_builtin(
    name = "barh",
    category = "plotting",
    summary = "Create horizontal bar charts.",
    keywords = "barh,horizontal bar,barchart,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::bar::BARH_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::bar"
)]
pub async fn barh_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    bar_impl(args, BARH_CONFIG).await
}

async fn bar_impl(args: Vec<Value>, config: BarRenderConfig) -> crate::BuiltinResult<f64> {
    let (x_values, values, rest) = parse_bar_call_args(args, config)?;
    let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
    let style = parse_bar_style_args(config.builtin_name, &rest, defaults)?;
    let mut input = Some(BarInput::from_value(x_values, values, config)?);
    let opts = PlotRenderOptions {
        title: config.title,
        x_label: config.x_label,
        y_label: config.y_label,
        ..Default::default()
    };
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(config.builtin_name, opts, move |figure, axes| {
        let style = style.clone();
        let arg = input.take().expect("bar input consumed once");
        if !style.requires_cpu_path() {
            if let Some(handle) = arg.gpu_handle() {
                match build_bar_gpu_series(arg.x_tensor().cloned(), handle, &style, config) {
                    Ok(charts) if !charts.is_empty() => {
                        let total = charts.len();
                        for (idx, mut bar) in charts.into_iter().enumerate() {
                            let default_label = default_series_label(idx, total);
                            apply_bar_style(&mut bar, &style, &default_label);
                            let plot_index = figure.add_bar_chart_on_axes(bar, axes);
                            if idx == 0 {
                                *plot_index_slot.borrow_mut() = Some((axes, plot_index));
                            }
                        }
                        return Ok(());
                    }
                    Ok(_) => {}
                    Err(err) => {
                        warn!("{} GPU path unavailable: {err}", config.builtin_name);
                    }
                }
            }
        }
        let (x_tensor, tensor) = arg.into_tensors(config)?;
        let charts = build_bar_series_from_tensor(x_tensor, tensor, &style, config)?;
        let total = charts.len();
        for (idx, mut bar) in charts.into_iter().enumerate() {
            let default_label = default_series_label(idx, total);
            apply_bar_style(&mut bar, &style, &default_label);
            let plot_index = figure.add_bar_chart_on_axes(bar, axes);
            if idx == 0 {
                *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            }
        }
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_bar_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

const DEFAULT_BAR_WIDTH: f32 = 0.75;

const BAR_DEFAULT_LABEL: &str = "Series 1";
const MATLAB_COLOR_ORDER: [Vec4; 7] = [
    Vec4::new(0.0, 0.447, 0.741, 1.0),
    Vec4::new(0.85, 0.325, 0.098, 1.0),
    Vec4::new(0.929, 0.694, 0.125, 1.0),
    Vec4::new(0.494, 0.184, 0.556, 1.0),
    Vec4::new(0.466, 0.674, 0.188, 1.0),
    Vec4::new(0.301, 0.745, 0.933, 1.0),
    Vec4::new(0.635, 0.078, 0.184, 1.0),
];

fn default_bar_color() -> Vec4 {
    Vec4::new(0.2, 0.6, 0.9, 0.95)
}

fn parse_bar_call_args(
    args: Vec<Value>,
    config: BarRenderConfig,
) -> BuiltinResult<(Option<Value>, Value, Vec<Value>)> {
    if args.is_empty() {
        return Err(config.invalid("expected at least one input"));
    }
    let mut it = args.into_iter();
    let first = it.next().expect("first");
    let Some(second) = it.next() else {
        return Ok((None, first, Vec::new()));
    };
    let trailing: Vec<Value> = it.collect();
    if matches!(second, Value::String(_) | Value::CharArray(_)) {
        let mut rest = vec![second];
        rest.extend(trailing);
        return Ok((None, first, rest));
    }
    if is_positional_bar_width_candidate(&second, &trailing) {
        let mut rest = vec![second];
        rest.extend(trailing);
        return Ok((None, first, rest));
    }
    Ok((Some(first), second, trailing))
}

fn is_positional_bar_width_candidate(second: &Value, trailing: &[Value]) -> bool {
    if !is_numeric_scalar_value(second) {
        return false;
    }
    trailing
        .first()
        .is_none_or(|value| matches!(value, Value::String(_) | Value::CharArray(_)))
}

fn is_numeric_scalar_value(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) => true,
        Value::Tensor(tensor) => tensor_utils::is_scalar_tensor(tensor),
        _ => false,
    }
}

fn build_bar_chart_with_x(
    x_tensor: Option<Tensor>,
    values: Vec<f64>,
    config: BarRenderConfig,
) -> BuiltinResult<BarChart> {
    if values.is_empty() {
        return Err(config.invalid("input cannot be empty"));
    }
    let labels = category_labels_from_x(x_tensor, values.len(), config)?;
    BarChart::new(labels, values)
        .map(|chart| chart.with_orientation(config.orientation))
        .map_err(|err| config.internal(&err))
}

fn build_bar_gpu_series(
    x_tensor: Option<Tensor>,
    values: &GpuTensorHandle,
    style: &BarStyle,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<BarChart>> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(config.builtin_name)?;
    let exported = runmat_accelerate_api::export_wgpu_buffer(values)
        .ok_or_else(|| config.internal("unable to export GPU values"))?;
    let x_count = x_tensor.as_ref().map(numeric_tensor_len);
    let shape = BarMatrixShape::from_handle(values, x_count, config)?;
    if shape.rows == 0 {
        return Err(config.invalid("input cannot be empty"));
    }
    if exported.len != shape.rows * shape.cols {
        return Err(config.invalid("gpuArray shape mismatch"));
    }
    let scalar = ScalarType::from_is_f64(exported.precision == ProviderPrecision::F64);
    let inputs = BarGpuInputs {
        values_buffer: exported.buffer.clone(),
        row_count: shape.rows as u32,
        scalar,
    };
    if shape.cols == 1 {
        let params = BarGpuParams {
            color: style.face_rgba(),
            bar_width: style.bar_width,
            series_index: 0,
            series_count: 1,
            source_row_count: shape.source_rows as u32,
            transpose_source: shape.transposed,
            group_index: 0,
            group_count: 1,
            orientation: config.gpu_orientation,
            layout: BarLayoutMode::Grouped,
        };
        let gpu_vertices = runmat_plot::gpu::bar::pack_vertices_from_values(
            &context.device,
            &context.queue,
            &inputs,
            &params,
        )
        .map_err(|e| config.internal(format!("failed to build GPU vertices: {e}")))?;
        let labels = category_labels_from_x(x_tensor, shape.rows, config)?;
        let bounds = build_bar_gpu_bounds(values, shape.rows, params.bar_width, config)?;
        let vertex_count = gpu_vertices.vertex_count;
        let chart = BarChart::from_gpu_buffer(
            labels,
            shape.rows,
            gpu_vertices,
            vertex_count,
            bounds,
            params.color,
            params.bar_width,
        )
        .with_orientation(config.orientation)
        .with_gpu_source_layout(inputs.clone(), 0, 1, shape.source_rows, shape.transposed);
        return Ok(vec![chart]);
    }
    build_bar_gpu_matrix_charts(&context, values, &inputs, x_tensor, shape, style, config)
}

fn build_bar_gpu_matrix_charts(
    context: &runmat_plot::SharedWgpuContext,
    values: &GpuTensorHandle,
    inputs: &BarGpuInputs,
    x_tensor: Option<Tensor>,
    shape: BarMatrixShape,
    style: &BarStyle,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<BarChart>> {
    let labels = category_labels_from_x(x_tensor, shape.rows, config)?;
    let layout_mode = match style.layout {
        BarLayout::Grouped => BarLayoutMode::Grouped,
        BarLayout::Stacked => BarLayoutMode::Stacked,
    };
    let bounds = if style.layout == BarLayout::Stacked {
        build_stacked_bar_gpu_bounds(values, shape, style.bar_width, config)?
    } else {
        build_bar_gpu_bounds(values, shape.rows, style.bar_width, config)?
    };
    let mut charts = Vec::with_capacity(shape.cols);
    for col in 0..shape.cols {
        let params = BarGpuParams {
            color: style.face_rgba(),
            bar_width: style.bar_width,
            series_index: col as u32,
            series_count: shape.cols as u32,
            source_row_count: shape.source_rows as u32,
            transpose_source: shape.transposed,
            group_index: if style.layout == BarLayout::Stacked {
                0
            } else {
                col as u32
            },
            group_count: if style.layout == BarLayout::Stacked {
                1
            } else {
                shape.cols as u32
            },
            orientation: config.gpu_orientation,
            layout: layout_mode,
        };
        let gpu_vertices = runmat_plot::gpu::bar::pack_vertices_from_values(
            &context.device,
            &context.queue,
            inputs,
            &params,
        )
        .map_err(|e| config.internal(format!("failed to build GPU vertices: {e}")))?;
        let vertex_count = gpu_vertices.vertex_count;
        let mut chart = BarChart::from_gpu_buffer(
            labels.clone(),
            shape.rows,
            gpu_vertices,
            vertex_count,
            bounds,
            params.color,
            params.bar_width,
        )
        .with_orientation(config.orientation)
        .with_gpu_source_layout(
            inputs.clone(),
            col,
            shape.cols,
            shape.source_rows,
            shape.transposed,
        );
        chart = if style.layout == BarLayout::Stacked {
            chart.with_group(0, 1)
        } else {
            chart.with_group(col, shape.cols)
        };
        charts.push(chart);
    }
    Ok(charts)
}

#[derive(Clone, Copy)]
struct BarMatrixShape {
    rows: usize,
    cols: usize,
    source_rows: usize,
    transposed: bool,
}

impl BarMatrixShape {
    fn from_handle(
        handle: &GpuTensorHandle,
        x_count: Option<usize>,
        config: BarRenderConfig,
    ) -> BuiltinResult<Self> {
        if handle.shape.is_empty() {
            return Err(config.invalid("input cannot be empty"));
        }
        if handle.shape.len() == 1 {
            return Ok(Self {
                rows: handle.shape[0],
                cols: 1,
                source_rows: handle.shape[0],
                transposed: false,
            });
        }
        if handle.shape.len() != 2 {
            return Err(config.invalid("matrix inputs must be 2-D"));
        }
        let rows = handle.shape[0];
        let cols = handle.shape[1];
        if rows == 0 || cols == 0 {
            return Err(config.invalid("input cannot be empty"));
        }
        if rows == 1 && cols > 1 {
            if x_count == Some(1) {
                return Ok(Self {
                    rows,
                    cols,
                    source_rows: rows,
                    transposed: false,
                });
            }
            return Ok(Self {
                rows: cols,
                cols: 1,
                source_rows: cols,
                transposed: false,
            });
        }
        if cols == 1 {
            return Ok(Self {
                rows,
                cols: 1,
                source_rows: rows,
                transposed: false,
            });
        }
        if let Some(count) = x_count {
            if count == rows {
                return Ok(Self {
                    rows,
                    cols,
                    source_rows: rows,
                    transposed: false,
                });
            }
            if count == cols {
                return Ok(Self {
                    rows: cols,
                    cols: rows,
                    source_rows: rows,
                    transposed: true,
                });
            }
        }
        Ok(Self {
            rows,
            cols,
            source_rows: rows,
            transposed: false,
        })
    }
}

fn numeric_tensor_len(tensor: &Tensor) -> usize {
    tensor_utils::tensor_element_len(tensor)
}

fn category_labels_from_x(
    x_tensor: Option<Tensor>,
    rows: usize,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<String>> {
    let Some(x) = x_tensor else {
        return Ok((1..=rows).map(|idx| format!("{idx}")).collect());
    };
    let labels: Vec<String> = numeric_vector(x)
        .into_iter()
        .map(format_number_label)
        .collect();
    if labels.len() != rows {
        return Err(config.invalid(format!(
            "X must contain {rows} category position{}",
            if rows == 1 { "" } else { "s" }
        )));
    }
    Ok(labels)
}

fn build_bar_gpu_bounds(
    values: &GpuTensorHandle,
    rows: usize,
    bar_width: f32,
    config: BarRenderConfig,
) -> BuiltinResult<BoundingBox> {
    let (min_value, max_value) = axis_bounds(values, config.builtin_name)?;
    let min_value = min_value.min(0.0);
    let max_value = max_value.max(0.0);
    let min_category = 1.0 - bar_width * 0.5;
    let max_category = rows as f32 + bar_width * 0.5;
    Ok(match config.orientation {
        Orientation::Vertical => BoundingBox::new(
            Vec3::new(min_category, min_value, 0.0),
            Vec3::new(max_category, max_value, 0.0),
        ),
        Orientation::Horizontal => BoundingBox::new(
            Vec3::new(min_value, min_category, 0.0),
            Vec3::new(max_value, max_category, 0.0),
        ),
    })
}

fn build_stacked_bar_gpu_bounds(
    values: &GpuTensorHandle,
    shape: BarMatrixShape,
    bar_width: f32,
    config: BarRenderConfig,
) -> BuiltinResult<BoundingBox> {
    let tensor = gather_tensor_from_gpu(values.clone(), config)?;
    let expected_len = shape.source_rows
        * if shape.transposed {
            shape.rows
        } else {
            shape.cols
        };
    if tensor_utils::tensor_element_len(&tensor) != expected_len {
        return Err(config.invalid("gpuArray shape mismatch"));
    }
    let tensor_values = tensor_utils::tensor_values_f64_cow(&tensor);
    let mut pos = vec![0.0f64; shape.rows];
    let mut neg = vec![0.0f64; shape.rows];
    let mut max_pos = 0.0f64;
    let mut min_neg = 0.0f64;
    for col in 0..shape.cols {
        for row in 0..shape.rows {
            let value_index = if shape.transposed {
                row * shape.source_rows + col
            } else {
                row + col * shape.source_rows
            };
            let value = tensor_values[value_index];
            if !value.is_finite() {
                continue;
            }
            if value >= 0.0 {
                pos[row] += value;
                if pos[row] > max_pos {
                    max_pos = pos[row];
                }
            } else {
                neg[row] += value;
                if neg[row] < min_neg {
                    min_neg = neg[row];
                }
            }
        }
    }
    let min_value = min_neg.min(0.0) as f32;
    let max_value = max_pos.max(0.0) as f32;
    let min_category = 1.0 - bar_width * 0.5;
    let max_category = shape.rows as f32 + bar_width * 0.5;
    Ok(match config.orientation {
        Orientation::Vertical => BoundingBox::new(
            Vec3::new(min_category, min_value, 0.0),
            Vec3::new(max_category, max_value, 0.0),
        ),
        Orientation::Horizontal => BoundingBox::new(
            Vec3::new(min_value, min_category, 0.0),
            Vec3::new(max_value, max_category, 0.0),
        ),
    })
}

enum BarInput {
    Host {
        x: Option<Tensor>,
        y: Tensor,
    },
    Gpu {
        x: Option<Tensor>,
        y: GpuTensorHandle,
    },
}

impl BarInput {
    fn from_value(x: Option<Value>, value: Value, config: BarRenderConfig) -> BuiltinResult<Self> {
        match (x, value) {
            (x, Value::GpuTensor(handle)) => {
                let x = match x {
                    Some(Value::GpuTensor(handle)) => Some(gather_tensor_from_gpu(handle, config)?),
                    Some(value) => Some(Tensor::try_from(&value).map_err(|e| config.invalid(&e))?),
                    None => None,
                };
                Ok(Self::Gpu { x, y: handle })
            }
            (x, other) => {
                let y = Tensor::try_from(&other).map_err(|e| config.invalid(&e))?;
                let x = match x {
                    Some(value) => Some(Tensor::try_from(&value).map_err(|e| config.invalid(&e))?),
                    None => None,
                };
                Ok(Self::Host { x, y })
            }
        }
    }

    fn gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu { y, .. } => Some(y),
            Self::Host { .. } => None,
        }
    }

    fn x_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Gpu { x, .. } | Self::Host { x, .. } => x.as_ref(),
        }
    }

    fn into_tensors(self, config: BarRenderConfig) -> BuiltinResult<(Option<Tensor>, Tensor)> {
        match self {
            Self::Host { x, y } => Ok((x, y)),
            Self::Gpu { x, y } => Ok((x, gather_tensor_from_gpu(y, config)?)),
        }
    }
}

async fn gather_tensor_from_gpu_async(
    handle: GpuTensorHandle,
    config: BarRenderConfig,
) -> BuiltinResult<Tensor> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, config.builtin_name))?;
    Tensor::try_from(&gathered)
        .map_err(|e| config.internal(format!("{}: {e}", config.builtin_name)))
}

fn gather_tensor_from_gpu(
    handle: GpuTensorHandle,
    config: BarRenderConfig,
) -> BuiltinResult<Tensor> {
    block_on(gather_tensor_from_gpu_async(handle, config))
}

pub(crate) fn apply_bar_style(bar: &mut BarChart, style: &BarStyle, default_label: &str) {
    if style.face_color_flat {
        let colors = generate_flat_colors(bar.bar_count(), style.face_alpha);
        bar.set_per_bar_colors(colors);
        bar.set_bar_width(style.bar_width);
    } else {
        bar.clear_per_bar_colors();
        let face = style.face_rgba();
        bar.apply_face_style(face, style.bar_width);
    }

    let outline = style.edge_rgba();
    bar.apply_outline_style(outline, style.line_width);

    if let Some(label) = &style.label {
        bar.label = Some(label.clone());
    } else if bar.label.is_none() {
        bar.label = Some(default_label.to_string());
    }
}

fn generate_flat_colors(count: usize, alpha: f32) -> Vec<Vec4> {
    let mut colors = Vec::with_capacity(count);
    if count == 0 {
        return colors;
    }
    for i in 0..count {
        let base = MATLAB_COLOR_ORDER[i % MATLAB_COLOR_ORDER.len()];
        colors.push(Vec4::new(base.x, base.y, base.z, alpha));
    }
    colors
}

fn default_series_label(index: usize, total: usize) -> String {
    if total <= 1 {
        BAR_DEFAULT_LABEL.to_string()
    } else {
        format!("Series {}", index + 1)
    }
}

enum BarTensorInput {
    Vector(Vec<f64>),
    Matrix(BarMatrixData),
}

struct BarMatrixData {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl BarMatrixData {
    fn value(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }

    fn transpose(self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for col in 0..self.rows {
            for row in 0..self.cols {
                data.push(self.value(col, row));
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }
}

fn tensor_to_bar_input(tensor: Tensor, config: BarRenderConfig) -> BuiltinResult<BarTensorInput> {
    if tensor.shape.is_empty() {
        return Err(config.invalid("input cannot be empty"));
    }
    if tensor.shape.len() == 1 || tensor.rows <= 1 || tensor.cols <= 1 {
        return Ok(BarTensorInput::Vector(numeric_vector(tensor)));
    }
    if tensor.shape.len() != 2 {
        return Err(config.invalid("matrix inputs must be 2-D"));
    }
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    if rows == 0 || cols == 0 {
        return Err(config.invalid("input cannot be empty"));
    }
    if rows * cols != tensor_utils::tensor_element_len(&tensor) {
        return Err(config.invalid("matrix inputs must be dense numeric arrays"));
    }
    let data = tensor_utils::tensor_values_f64(&tensor);
    Ok(BarTensorInput::Matrix(BarMatrixData { rows, cols, data }))
}

fn build_bar_series_from_tensor(
    x_tensor: Option<Tensor>,
    tensor: Tensor,
    style: &BarStyle,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<BarChart>> {
    match tensor_to_bar_input(tensor, config)? {
        BarTensorInput::Vector(values) => {
            if x_tensor
                .as_ref()
                .is_some_and(|x| numeric_tensor_len(x) == 1)
                && values.len() > 1
            {
                return build_bar_series_from_matrix(
                    x_tensor,
                    BarMatrixData {
                        rows: 1,
                        cols: values.len(),
                        data: values,
                    },
                    style,
                    config,
                );
            }
            let bar = build_bar_chart_with_x(x_tensor, values, config)?;
            Ok(vec![bar])
        }
        BarTensorInput::Matrix(matrix) => {
            build_bar_series_from_matrix(x_tensor, matrix, style, config)
        }
    }
}

fn build_bar_series_from_matrix(
    x_tensor: Option<Tensor>,
    matrix: BarMatrixData,
    style: &BarStyle,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<BarChart>> {
    let x_len = x_tensor.as_ref().map(numeric_tensor_len);
    let matrix = if x_len == Some(matrix.cols) && matrix.rows != matrix.cols {
        matrix.transpose()
    } else {
        matrix
    };
    if matrix.cols == 0 {
        return Err(config.invalid("input cannot be empty"));
    }
    let labels = category_labels_from_x(x_tensor, matrix.rows, config)?;
    let mut charts = Vec::with_capacity(matrix.cols);
    let mut pos_offsets = vec![0.0f64; matrix.rows];
    let mut neg_offsets = vec![0.0f64; matrix.rows];
    for col in 0..matrix.cols {
        let mut values = Vec::with_capacity(matrix.rows);
        for row in 0..matrix.rows {
            values.push(matrix.value(row, col));
        }
        let mut chart = BarChart::new(labels.clone(), values.clone())
            .map(|chart| chart.with_orientation(config.orientation))
            .map_err(|err| config.internal(&err))?;
        if style.layout == BarLayout::Stacked {
            let offsets = compute_stack_offsets(&values, &mut pos_offsets, &mut neg_offsets);
            chart = chart.with_stack_offsets(offsets).with_group(0, 1);
        } else {
            chart = chart.with_group(col, matrix.cols);
        }
        charts.push(chart);
    }
    Ok(charts)
}

fn format_number_label(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

fn compute_stack_offsets(
    values: &[f64],
    pos_offsets: &mut [f64],
    neg_offsets: &mut [f64],
) -> Vec<f64> {
    let mut offsets = vec![0.0f64; values.len()];
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            offsets[idx] = 0.0;
            continue;
        }
        if value >= 0.0 {
            offsets[idx] = pos_offsets[idx];
            pos_offsets[idx] += value;
        } else {
            offsets[idx] = neg_offsets[idx];
            neg_offsets[idx] += value;
        }
    }
    offsets
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Value};
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn bar_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
        block_on(super::bar_builtin(args))
    }

    fn barh_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
        block_on(super::barh_builtin(args))
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("bar test vector")
    }

    fn matrix_tensor(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).expect("bar test matrix")
    }

    fn poisoned_integer_matrix(storage: IntegerStorage, rows: usize, cols: usize) -> Tensor {
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer matrix");
        tensor.data.clear();
        tensor
    }

    fn row_vector_tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![1, data.len()]).expect("bar test row vector")
    }

    fn gpu_handle_with_shape(shape: &[usize]) -> GpuTensorHandle {
        GpuTensorHandle {
            shape: shape.to_vec(),
            device_id: 0,
            buffer_id: 0,
        }
    }

    #[test]
    fn bar_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = BAR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = bar(Y)"));
        assert!(labels.contains(&"h = bar(Y, width)"));
        assert!(labels.contains(&"h = bar(Y, width, color)"));
        assert!(labels.contains(&"h = bar(X, Y)"));
        assert!(labels.contains(&"h = bar(X, Y, width)"));
        assert!(labels.contains(&"h = bar(X, Y, Name, Value, ...)"));
    }

    #[test]
    fn barh_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = BARH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = barh(Y)"));
        assert!(labels.contains(&"h = barh(Y, width)"));
        assert!(labels.contains(&"h = barh(Y, width, color)"));
        assert!(labels.contains(&"h = barh(X, Y)"));
        assert!(labels.contains(&"h = barh(X, Y, width)"));
        assert!(labels.contains(&"h = barh(X, Y, Name, Value, ...)"));
    }

    #[test]
    fn bar_invalid_argument_uses_stable_identifier() {
        let err = block_on(super::bar_builtin(vec![])).expect_err("missing args should fail");
        assert_eq!(err.identifier(), BAR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn barh_invalid_argument_uses_stable_identifier() {
        let err = block_on(super::barh_builtin(vec![])).expect_err("missing args should fail");
        assert_eq!(err.identifier(), BARH_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_requires_non_empty_input() {
        setup_plot_tests();
        assert!(build_bar_chart_with_x(None, vec![], BAR_CONFIG).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_builtin_matches_backend_contract() {
        setup_plot_tests();
        let out = bar_builtin(vec![Value::Tensor(tensor_from(&[1.0, 2.0, 3.0]))]);
        if let Err(err) = out {
            let msg_lower = err.to_string().to_lowercase();
            assert!(
                msg_lower.contains("plotting is unavailable")
                    || msg_lower.contains("non-main thread"),
                "unexpected error: {err}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn barh_builtin_matches_backend_contract() {
        setup_plot_tests();
        let out = barh_builtin(vec![Value::Tensor(tensor_from(&[1.0, 2.0, 3.0]))]);
        if let Err(err) = out {
            let msg_lower = err.to_string().to_lowercase();
            assert!(
                msg_lower.contains("plotting is unavailable")
                    || msg_lower.contains("non-main thread"),
                "unexpected error: {err}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_parser_handles_stacked_flag() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("bar", &[Value::String("stacked".into())], defaults).unwrap();
        assert_eq!(style.layout, BarLayout::Stacked);
    }

    #[test]
    fn bar_parser_treats_positional_width_as_style() {
        let (x, y, rest) = parse_bar_call_args(
            vec![
                Value::Tensor(row_vector_tensor(&[10.0, 22.0, 30.0, 42.0])),
                Value::from(0.4),
                Value::String("red".into()),
            ],
            BARH_CONFIG,
        )
        .unwrap();
        assert!(x.is_none());
        assert!(matches!(y, Value::Tensor(_)));
        assert_eq!(rest.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_grouped() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let tensor = matrix_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(None, tensor, &style, BAR_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_reads_typed_integer_storage() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let tensor = poisoned_integer_matrix(IntegerStorage::I16(vec![1, -2, 3, 4]), 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BAR_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
        assert_eq!(charts[0].values().unwrap(), &[1.0, -2.0]);
        assert_eq!(charts[1].values().unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn bar_width_candidate_accepts_typed_integer_scalar_without_double_mirror() {
        let mut width =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("typed width");
        width.data.clear();

        assert!(is_positional_bar_width_candidate(
            &Value::Tensor(width),
            &[Value::String("grouped".into())]
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_stacked() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("bar", &[Value::String("stacked".into())], defaults).unwrap();
        let tensor = matrix_tensor(&[1.0, -2.0, 3.0, 4.0], 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BAR_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn barh_series_from_matrix_sets_horizontal_orientation() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("barh", &[], defaults).unwrap();
        let tensor = matrix_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BARH_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
        assert!(charts
            .iter()
            .all(|chart| chart.orientation == Orientation::Horizontal));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn barh_stacked_series_preserves_horizontal_offsets() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("barh", &[Value::String("stacked".into())], defaults).unwrap();
        let tensor = matrix_tensor(&[1.0, -2.0, 3.0, 4.0], 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BARH_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
        assert!(charts
            .iter()
            .all(|chart| chart.orientation == Orientation::Horizontal));
        assert_eq!(charts[1].stack_offsets().unwrap(), &[1.0, 0.0]);
    }

    #[test]
    fn bar_supports_explicit_x_values() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(tensor_from(&[10.0, 20.0])),
            tensor_from(&[1.0, 2.0]),
            &style,
            BAR_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 1);
        assert_eq!(charts[0].labels[0], "10");
        assert_eq!(charts[0].labels[1], "20");
    }

    #[test]
    fn barh_treats_row_vectors_as_vector_inputs() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("barh", &[Value::String("stacked".into())], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(row_vector_tensor(&[1980.0, 1990.0, 2000.0])),
            row_vector_tensor(&[10.0, 20.0, 30.0]),
            &style,
            BARH_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 1);
        assert_eq!(charts[0].bar_count(), 3);
        assert_eq!(charts[0].orientation, Orientation::Horizontal);
    }

    #[test]
    fn barh_treats_scalar_x_vector_y_as_single_group() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("barh", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(tensor_from(&[1990.0])),
            row_vector_tensor(&[10.0, 20.0, 30.0]),
            &style,
            BARH_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 3);
        assert_eq!(charts[0].bar_count(), 1);
        assert_eq!(charts[0].labels[0], "1990");
        assert!(charts
            .iter()
            .all(|chart| chart.orientation == Orientation::Horizontal));
    }

    #[test]
    fn barh_supports_x_matching_matrix_columns() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("barh", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(row_vector_tensor(&[1980.0, 1990.0, 2000.0])),
            matrix_tensor(&[2.0, 11.0, 6.0, 22.0, 9.0, 32.0], 2, 3),
            &style,
            BARH_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 2);
        assert_eq!(charts[0].bar_count(), 3);
        assert_eq!(charts[0].values().unwrap(), &[2.0, 6.0, 9.0]);
        assert_eq!(charts[1].values().unwrap(), &[11.0, 22.0, 32.0]);
        assert_eq!(charts[0].labels, vec!["1980", "1990", "2000"]);
    }

    #[test]
    fn gpu_shape_matches_host_row_vector_semantics() {
        let implicit =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[1, 3]), None, BARH_CONFIG)
                .unwrap();
        assert_eq!(implicit.rows, 3);
        assert_eq!(implicit.cols, 1);

        let scalar_x =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[1, 3]), Some(1), BARH_CONFIG)
                .unwrap();
        assert_eq!(scalar_x.rows, 1);
        assert_eq!(scalar_x.cols, 3);

        let vector_x =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[1, 3]), Some(3), BARH_CONFIG)
                .unwrap();
        assert_eq!(vector_x.rows, 3);
        assert_eq!(vector_x.cols, 1);

        let x_matches_columns =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[2, 3]), Some(3), BARH_CONFIG)
                .unwrap();
        assert_eq!(x_matches_columns.rows, 3);
        assert_eq!(x_matches_columns.cols, 2);
        assert_eq!(x_matches_columns.source_rows, 2);
        assert!(x_matches_columns.transposed);
    }

    #[test]
    fn explicit_x_with_gpu_y_preserves_gpu_input() {
        let input = BarInput::from_value(
            Some(Value::Tensor(tensor_from(&[1980.0, 1990.0, 2000.0]))),
            Value::GpuTensor(gpu_handle_with_shape(&[1, 3])),
            BARH_CONFIG,
        )
        .unwrap();
        assert!(input.gpu_handle().is_some());
        assert_eq!(input.x_tensor().map(numeric_tensor_len), Some(3));
    }

    #[test]
    fn bar_x_y_shorthand_builds_chart_with_explicit_labels() {
        setup_plot_tests();
        let out = bar_builtin(vec![
            Value::Tensor(tensor_from(&[10.0, 20.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
        ]);
        if let Err(err) = out {
            let msg = err.to_string().to_lowercase();
            assert!(msg.contains("plotting is unavailable") || msg.contains("non-main thread"));
        }
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(tensor_from(&[10.0, 20.0])),
            tensor_from(&[1.0, 2.0]),
            &style,
            BAR_CONFIG,
        )
        .unwrap();
        assert_eq!(charts[0].labels[0], "10");
        assert_eq!(charts[0].labels[1], "20");
    }

    #[test]
    fn bar_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}
