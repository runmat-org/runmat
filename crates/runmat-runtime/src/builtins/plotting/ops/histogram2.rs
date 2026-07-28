use std::cell::RefCell;
use std::rc::Rc;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_handle};
use crate::builtins::plotting::state::{
    color_limits_snapshot, render_active_plot, PlotRenderOptions,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "histogram2";
const DEFAULT_FACE_ALPHA: f64 = 1.0;

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created histogram2 chart.",
}];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-axis input samples.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y-axis input samples.",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const INPUTS_XY: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];

const INPUTS_XY_BINS: [BuiltinParamDescriptor; 3] = [
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar/two-element bin count or edge specification.",
    },
];

const INPUTS_XY_BINXY: [BuiltinParamDescriptor; 4] = [
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "binsX",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin count scalar or edge vector.",
    },
    BuiltinParamDescriptor {
        name: "binsY",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin count scalar or edge vector.",
    },
];

const INPUTS_XY_OPTIONS: [BuiltinParamDescriptor; 3] = [
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Binning and chart name/value options.",
    },
];

const INPUTS_AX_XY: [BuiltinParamDescriptor; 3] = [PARAM_AX, PARAM_X, PARAM_Y];

const INPUTS_AX_XY_BINS: [BuiltinParamDescriptor; 4] = [
    PARAM_AX,
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar/two-element bin count or edge specification.",
    },
];

const INPUTS_AX_XY_BINXY: [BuiltinParamDescriptor; 5] = [
    PARAM_AX,
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "binsX",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin count scalar or edge vector.",
    },
    BuiltinParamDescriptor {
        name: "binsY",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin count scalar or edge vector.",
    },
];

const INPUTS_AX_XY_OPTIONS: [BuiltinParamDescriptor; 4] = [
    PARAM_AX,
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Binning and chart name/value options.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y)",
        inputs: &INPUTS_XY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y, bins)",
        inputs: &INPUTS_XY_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y, binsX, binsY)",
        inputs: &INPUTS_XY_BINXY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y, Name, Value, ...)",
        inputs: &INPUTS_XY_OPTIONS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y)",
        inputs: &INPUTS_AX_XY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y, bins)",
        inputs: &INPUTS_AX_XY_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y, binsX, binsY)",
        inputs: &INPUTS_AX_XY_BINXY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y, Name, Value, ...)",
        inputs: &INPUTS_AX_XY_OPTIONS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(X, Y)",
        inputs: &INPUTS_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(X, Y, bins)",
        inputs: &INPUTS_XY_BINS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(ax, X, Y)",
        inputs: &INPUTS_AX_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(ax, X, Y, Name, Value, ...)",
        inputs: &INPUTS_AX_XY_OPTIONS,
        outputs: &[],
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM2.INVALID_ARGUMENT",
    identifier: Some("RunMat:histogram2:InvalidArgument"),
    when: "Input arrays, bin controls, or chart options are malformed.",
    message: "histogram2: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM2.INTERNAL",
    identifier: Some("RunMat:histogram2:Internal"),
    when: "RunMat cannot compute bins, register state, or render the chart.",
    message: "histogram2: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const HISTOGRAM2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Histogram2DisplayStyle {
    Bar3,
    Tile,
}

impl Histogram2DisplayStyle {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bar3 => "bar3",
            Self::Tile => "tile",
        }
    }
}

#[derive(Clone, Debug)]
struct Histogram2Options {
    histcounts_args: Vec<Value>,
    display_style: Histogram2DisplayStyle,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<String>,
    normalization: String,
}

pub(crate) struct Histogram2Chart {
    pub(crate) values: Tensor,
    pub(crate) raw_counts: Tensor,
    pub(crate) x_bin_edges: Vec<f64>,
    pub(crate) y_bin_edges: Vec<f64>,
    pub(crate) surface: SurfacePlot,
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INTERNAL, message)
}

fn map_plot_error(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        invalid(err.message)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::histogram2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histogram2",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Omit,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "histogram2 is a plotting sink built on histcounts2; current builds materialise the 2-D bin grid before rendering.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::histogram2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histogram2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "histogram2 terminates fusion graphs and produces plotting output.",
};

#[runtime_builtin(
    name = "histogram2",
    category = "plotting",
    summary = "Create two-dimensional histogram charts.",
    keywords = "histogram2,histcounts2,2d histogram,density,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::histogram2::HISTOGRAM2_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::histogram2"
)]
pub async fn histogram2_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let (axes_target, args) = split_leading_axes_handle(args, NAME).map_err(map_plot_error)?;
    apply_axes_target(axes_target, NAME).map_err(map_plot_error)?;

    if args.len() < 2 {
        return Err(invalid("histogram2: expected X and Y inputs"));
    }
    let mut args = args.into_iter();
    let x = args.next().expect("x");
    let y = args.next().expect("y");
    let rest: Vec<Value> = args.collect();
    let options = parse_options(rest)?;

    let chart = build_chart(x.clone(), y.clone(), &options).await?;
    render_histogram2(options, chart)
}

fn parse_options(args: Vec<Value>) -> BuiltinResult<Histogram2Options> {
    let mut histcounts_args = Vec::new();
    let mut display_style = Histogram2DisplayStyle::Bar3;
    let mut show_empty_bins = true;
    let mut face_alpha = DEFAULT_FACE_ALPHA;
    let mut display_name = None;
    let mut normalization = "count".to_string();
    let mut index = 0usize;

    while index < args.len() {
        if let Some(key) = value_as_string(&args[index]) {
            if index + 1 >= args.len() {
                return Err(invalid(format!(
                    "histogram2: missing value for option '{key}'"
                )));
            }
            let value = &args[index + 1];
            match key.trim().to_ascii_lowercase().as_str() {
                "displaystyle" => {
                    display_style = parse_display_style(value)?;
                }
                "showemptybins" => {
                    show_empty_bins = option_bool(value, "ShowEmptyBins")?;
                }
                "facealpha" => {
                    face_alpha = option_scalar(value, "FaceAlpha")?;
                    validate_face_alpha(face_alpha)?;
                }
                "displayname" => {
                    display_name = Some(
                        tensor::value_to_string(value)
                            .ok_or_else(|| invalid("histogram2: DisplayName must be text"))?,
                    );
                }
                "normalization" => {
                    normalization = tensor::value_to_string(value)
                        .ok_or_else(|| invalid("histogram2: Normalization must be text"))?
                        .trim()
                        .to_ascii_lowercase();
                    histcounts_args.push(args[index].clone());
                    histcounts_args.push(value.clone());
                }
                "numbins" | "xbinwidth" | "ybinwidth" | "binwidth" | "xbinlimits"
                | "ybinlimits" | "binlimits" | "xbinedges" | "ybinedges" | "binmethod"
                | "xbinmethod" | "ybinmethod" => {
                    histcounts_args.push(args[index].clone());
                    histcounts_args.push(value.clone());
                }
                "facecolor" | "edgecolor" | "edgealpha" | "linestyle" => {
                    // Accepted as rendering style compatibility; RunMat's SurfacePlot backend maps
                    // histogram2 through colormapped surfaces and does not yet expose per-bin edges.
                }
                other => {
                    return Err(invalid(format!(
                        "histogram2: unrecognised option '{other}'"
                    )));
                }
            }
            index += 2;
        } else {
            histcounts_args.push(args[index].clone());
            index += 1;
        }
    }

    validate_normalization(&normalization)?;
    Ok(Histogram2Options {
        histcounts_args,
        display_style,
        show_empty_bins,
        face_alpha,
        display_name,
        normalization,
    })
}

async fn build_chart(
    x: Value,
    y: Value,
    options: &Histogram2Options,
) -> BuiltinResult<Histogram2Chart> {
    let raw_args = histcounts_count_args(&options.histcounts_args);
    let raw_eval = crate::builtins::stats::hist::histcounts2::evaluate(x, y, &raw_args)
        .await
        .map_err(map_histcounts_error)?;
    let (raw_counts_value, x_edges_value, y_edges_value) = raw_eval.into_triple();
    let raw_counts = Tensor::try_from(&raw_counts_value)
        .map_err(|err| invalid(format!("histogram2: cannot convert raw bin counts: {err}")))?;
    let x_edges = Tensor::try_from(&x_edges_value).map_err(|err| {
        invalid(format!(
            "histogram2: cannot convert XBinEdges tensor: {err}"
        ))
    })?;
    let y_edges = Tensor::try_from(&y_edges_value).map_err(|err| {
        invalid(format!(
            "histogram2: cannot convert YBinEdges tensor: {err}"
        ))
    })?;
    let values = apply_normalization_2d(
        &raw_counts,
        &x_edges.data,
        &y_edges.data,
        &options.normalization,
    )?;

    let surface = build_surface_from_values(
        &values,
        &x_edges.data,
        &y_edges.data,
        options.display_style,
        options.show_empty_bins,
        options.face_alpha,
        options.display_name.as_deref(),
        color_limits_snapshot(),
    )?;
    Ok(Histogram2Chart {
        values,
        raw_counts,
        x_bin_edges: x_edges.data,
        y_bin_edges: y_edges.data,
        surface,
    })
}

fn map_histcounts_error(err: RuntimeError) -> RuntimeError {
    invalid(err.message)
}

fn histcounts_count_args(args: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if value_as_string(&args[index])
            .is_some_and(|key| key.trim().eq_ignore_ascii_case("Normalization"))
        {
            index += 2;
        } else {
            out.push(args[index].clone());
            index += 1;
        }
    }
    out
}

pub(crate) fn build_surface_from_values(
    values: &Tensor,
    x_edges: &[f64],
    y_edges: &[f64],
    display_style: Histogram2DisplayStyle,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<&str>,
    color_limits: Option<(f64, f64)>,
) -> BuiltinResult<SurfacePlot> {
    if values.shape.len() < 2 || x_edges.len() < 2 || y_edges.len() < 2 {
        return Err(internal("histogram2: invalid 2-D histogram grid"));
    }
    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    if values.data.len() != x_bins * y_bins {
        return Err(internal(
            "histogram2: Values shape does not match bin edges",
        ));
    }
    validate_face_alpha(face_alpha)?;

    let mut z_grid = vec![vec![0.0; y_bins]; x_bins];
    for (ix, row) in z_grid.iter_mut().enumerate().take(x_bins) {
        for (iy, cell) in row.iter_mut().enumerate().take(y_bins) {
            let mut value = values.data[ix + iy * x_bins];
            if !show_empty_bins && value == 0.0 {
                value = f64::NAN;
            }
            *cell = value;
        }
    }

    let mut surface = SurfacePlot::new(bin_centers(x_edges), bin_centers(y_edges), z_grid)
        .map_err(|err| internal(format!("histogram2: {err}")))?
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::None)
        .with_alpha(face_alpha as f32);
    if display_style == Histogram2DisplayStyle::Tile {
        surface = surface.with_flatten_z(true).with_image_mode(true);
    }
    if let Some(display_name) = display_name {
        surface = surface.with_label(display_name.to_string());
    }
    if color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }
    Ok(surface)
}

fn render_histogram2(options: Histogram2Options, chart: Histogram2Chart) -> BuiltinResult<f64> {
    let mut surface = Some(chart.surface);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "Histogram2",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        move |figure, axes| {
            let plot_index = figure.add_surface_plot_on_axes(
                surface.take().expect("histogram2 plot consumed once"),
                axes,
            );
            figure.set_axes_colorbar_enabled(axes, true);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_histogram2_handle(
        figure_handle,
        axes,
        plot_index,
        chart.values,
        chart.raw_counts,
        chart.x_bin_edges,
        chart.y_bin_edges,
        options.normalization,
        options.display_style,
        options.show_empty_bins,
        options.face_alpha,
        options.display_name,
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(internal(err.message));
    }
    Ok(handle)
}

pub(crate) fn apply_normalization_2d(
    raw_counts: &Tensor,
    x_edges: &[f64],
    y_edges: &[f64],
    norm: &str,
) -> BuiltinResult<Tensor> {
    validate_normalization(norm)?;
    let x_bins = x_edges.len().saturating_sub(1);
    let y_bins = y_edges.len().saturating_sub(1);
    let counts = &raw_counts.data;
    let total: f64 = counts.iter().sum();
    let x_widths: Vec<f64> = x_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let y_widths: Vec<f64> = y_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let values = match norm {
        "count" => counts.clone(),
        "probability" => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        "countdensity" => density_counts(counts, &x_widths, &y_widths, x_bins, y_bins, 1.0),
        "pdf" | "probabilitydensity" => {
            if total > 0.0 {
                density_counts(counts, &x_widths, &y_widths, x_bins, y_bins, total)
            } else {
                vec![0.0; counts.len()]
            }
        }
        "cumcount" => cumulative_counts(counts, None),
        "cdf" => cumulative_counts(counts, Some(total)),
        other => {
            return Err(invalid(format!(
                "histogram2: unsupported normalization '{other}'"
            )))
        }
    };
    Tensor::new(values, raw_counts.shape.clone())
        .map_err(|err| internal(format!("histogram2: {err}")))
}

fn density_counts(
    counts: &[f64],
    x_widths: &[f64],
    y_widths: &[f64],
    x_bins: usize,
    y_bins: usize,
    total_scale: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; counts.len()];
    for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
        for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
            let idx = ix + iy * x_bins;
            let area = x_width * y_width;
            out[idx] = if area > 0.0 {
                counts[idx] / (total_scale * area)
            } else {
                0.0
            };
        }
    }
    out
}

fn cumulative_counts(counts: &[f64], total: Option<f64>) -> Vec<f64> {
    let mut acc = 0.0;
    counts
        .iter()
        .map(|&count| {
            acc += count;
            if let Some(total) = total {
                if total > 0.0 {
                    acc / total
                } else {
                    0.0
                }
            } else {
                acc
            }
        })
        .collect()
}

pub(crate) fn validate_normalization(norm: &str) -> BuiltinResult<()> {
    match norm.trim().to_ascii_lowercase().as_str() {
        "count" | "probability" | "countdensity" | "pdf" | "probabilitydensity" | "cumcount"
        | "cdf" => Ok(()),
        other => Err(invalid(format!(
            "histogram2: unsupported normalization '{other}'"
        ))),
    }
}

pub(crate) fn validate_face_alpha(value: f64) -> BuiltinResult<()> {
    if !(0.0..=1.0).contains(&value) || !value.is_finite() {
        return Err(invalid(
            "histogram2: FaceAlpha must be in the interval [0,1]",
        ));
    }
    Ok(())
}

pub(crate) fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => numeric_bool(*value, name),
        Value::Int(value) => numeric_bool(value.to_f64(), name),
        _ => {
            if let Some(text) = tensor::value_to_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" => Ok(true),
                    "off" | "false" | "0" => Ok(false),
                    _ => Err(invalid(format!("histogram2: {name} must be logical"))),
                }
            } else {
                Err(invalid(format!("histogram2: {name} must be logical")))
            }
        }
    }
}

fn numeric_bool(value: f64, name: &str) -> BuiltinResult<bool> {
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(invalid(format!("histogram2: {name} must be 0 or 1")))
    }
}

pub(crate) fn option_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        _ => Err(invalid(format!("histogram2: {name} must be a scalar"))),
    }
}

pub(crate) fn parse_display_style(value: &Value) -> BuiltinResult<Histogram2DisplayStyle> {
    let text = tensor::value_to_string(value)
        .ok_or_else(|| invalid("histogram2: DisplayStyle must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "bar3" => Ok(Histogram2DisplayStyle::Bar3),
        "tile" => Ok(Histogram2DisplayStyle::Tile),
        other => Err(invalid(format!(
            "histogram2: unsupported DisplayStyle '{other}'"
        ))),
    }
}

fn bin_centers(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

fn value_as_string(value: &Value) -> Option<String> {
    tensor::value_to_string(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::figure::PlotElement;

    fn vec_tensor(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    #[test]
    fn histogram2_counts_edges_and_registers_chart_state() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(values.data, vec![2.0, 0.0, 0.0, 2.0]);

        let figure = clone_figure(current_figure_handle()).unwrap();
        let plot = figure.plots().next().unwrap();
        let PlotElement::Surface(surface) = plot else {
            panic!("expected surface plot");
        };
        assert!(!surface.image_mode);
        assert!(!surface.flatten_z);
        assert_eq!(surface.x_data, vec![0.25, 0.75]);
        assert_eq!(surface.y_data, vec![0.25, 0.75]);
    }

    #[test]
    fn histogram2_supports_tile_style_and_normalization() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::String("NumBins".into()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
            Value::String("Normalization".into()),
            Value::String("probability".into()),
            Value::String("DisplayStyle".into()),
            Value::String("tile".into()),
            Value::String("ShowEmptyBins".into()),
            Value::String("off".into()),
            Value::String("FaceAlpha".into()),
            Value::Num(0.5),
            Value::String("DisplayName".into()),
            Value::String("density".into()),
        ]))
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(values.data, vec![0.5, 0.0, 0.0, 0.5]);
        assert!(matches!(
            get_builtin(vec![Value::Num(handle), Value::String("DisplayStyle".into())]).unwrap(),
            Value::String(style) if style == "tile"
        ));
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert!(surface.image_mode);
        assert!(surface.flatten_z);
        assert_eq!(surface.alpha, 0.5);
        assert_eq!(surface.label.as_deref(), Some("density"));
    }

    #[test]
    fn histogram2_set_updates_display_and_normalization_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("Normalization".into()),
            Value::String("cdf".into()),
            Value::String("DisplayStyle".into()),
            Value::String("tile".into()),
            Value::String("ShowEmptyBins".into()),
            Value::String("off".into()),
        ])
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(values.data, vec![0.5, 0.5, 0.5, 1.0]);
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert!(surface.image_mode);
    }

    #[test]
    fn histogram2_rejects_nonscalar_face_alpha_updates() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let err = set_builtin(vec![
            Value::Num(handle),
            Value::String("FaceAlpha".into()),
            Value::Tensor(Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap()),
        ])
        .expect_err("non-scalar FaceAlpha should be rejected");
        assert!(err.message.contains("FaceAlpha must be a scalar"));
    }
}
