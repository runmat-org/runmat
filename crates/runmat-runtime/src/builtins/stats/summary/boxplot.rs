//! Box plot compatibility helper.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use glam::Vec4;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Type,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{LinePlot, LineStyle, MarkerStyle, ScatterPlot};
use runmat_value::{CellArray, CharArray, IntValue, NumericScalar, Tensor, Value};

use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_handle};
use crate::builtins::plotting::state::{render_active_plot, PlotRenderOptions};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "boxplot";

const OUTPUT_HANDLES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of graphics handles for created box plot primitives.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric vector or matrix data.",
};

const PARAM_G: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "g",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Grouping vector or cell array of grouping vectors.",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Box plot name-value options.",
};

const INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUT_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_OPTIONS];
const INPUT_X_G: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_G];
const INPUT_X_G_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_G, PARAM_OPTIONS];
const INPUT_AX_X: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_X];
const INPUT_AX_X_G_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_AX, PARAM_X, PARAM_G, PARAM_OPTIONS];
const OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLES];

const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "boxplot(x)",
        inputs: &INPUT_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(x, g)",
        inputs: &INPUT_X_G,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(___, Name, Value)",
        inputs: &INPUT_X_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(ax, ___)",
        inputs: &INPUT_AX_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "boxplot(ax, x, g, Name, Value)",
        inputs: &INPUT_AX_X_G_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = boxplot(___)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = boxplot(x, g, Name, Value)",
        inputs: &INPUT_X_G_OPTIONS,
        outputs: &OUTPUT_H,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOXPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:boxplot:InvalidArgument"),
    when: "Inputs, grouping variables, or name-value options are malformed.",
    message: "boxplot: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOXPLOT.INTERNAL",
    identifier: Some("RunMat:boxplot:Internal"),
    when: "RunMat cannot construct box statistics or plotting primitives.",
    message: "boxplot: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

const BOXPLOT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotIntegerDataExtension"),
};

const BOXPLOT_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with logical sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotLogicalDataExtension"),
};

const BOXPLOT_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-integer-group",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with integer grouping data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotIntegerGroupExtension"),
};

const BOXPLOT_LOGICAL_GROUP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-logical-group",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with logical grouping data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotLogicalGroupExtension"),
};

const BOXPLOT_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with an integer numeric option is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotIntegerOptionExtension"),
};

const BOXPLOT_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-logical-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with a logical numeric option is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotLogicalOptionExtension"),
};

const BOXPLOT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "boxplot-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "boxplot with gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BoxplotGpuInputExtension"),
};

const BOXPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    BOXPLOT_INTEGER_DATA_EXTENSION,
    BOXPLOT_LOGICAL_DATA_EXTENSION,
    BOXPLOT_INTEGER_GROUP_EXTENSION,
    BOXPLOT_LOGICAL_GROUP_EXTENSION,
    BOXPLOT_INTEGER_OPTION_EXTENSION,
    BOXPLOT_LOGICAL_OPTION_EXTENSION,
    BOXPLOT_GPU_INPUT_EXTENSION,
];

const BOXPLOT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every integer class as sample data; the public sample-data domain is single or double.",
    }];

const BOXPLOT_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "g or ColorGroup",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every integer class as direct or cell-contained grouping data and retains exact integer labels; the public numeric grouping domain is single or double.",
    }];

const BOXPLOT_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric Name-Value option",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts integer numeric option values; visualization controls cross an explicit floating boundary while numeric labels and grouping options retain exact text or identity. Public numeric option domains are single or double.",
    }];

pub const BOXPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "boxplot(x) with integer sample data",
        inputs: &BOXPLOT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only visualization form; integer observations cross one deliberate binary64 statistics/rendering boundary and the optional result is a host graphics-handle vector.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "boxplot(x,g) or boxplot(___,ColorGroup,g) with integer grouping data",
        inputs: &BOXPLOT_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only grouping form; integer group identity, order, and labels remain exact for every supported class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "boxplot(___,Name,Value) with integer numeric option values",
        inputs: &BOXPLOT_INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only option form; geometry, color, limit, and scale controls cross one deliberate binary64 boundary after option-specific validation, while labels and grouping identity remain exact.",
    },
];

pub const BOXPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Orientation {
    Vertical,
    Horizontal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExtremeMode {
    Clip,
    Compress,
}

#[derive(Clone, Debug)]
struct BoxplotOptions {
    whisker: f64,
    orientation: Orientation,
    labels: Option<Vec<String>>,
    group_order: Option<Vec<String>>,
    positions: Option<Vec<f64>>,
    widths: Vec<f64>,
    outlier_marker: Option<MarkerStyle>,
    outlier_color: Option<Vec4>,
    outlier_size: f32,
    box_colors: Vec<Vec4>,
    color_group: Option<Value>,
    data_lim: Option<(f64, f64)>,
    extreme_mode: ExtremeMode,
    jitter: f64,
    compact: bool,
}

#[derive(Clone, Debug)]
struct BoxStats {
    label: String,
    position: f64,
    width: f64,
    color: Vec4,
    q1: f64,
    median: f64,
    q3: f64,
    lower_whisker: f64,
    upper_whisker: f64,
    outliers: Vec<f64>,
}

#[derive(Clone, Debug)]
enum PlotPlan {
    Line(LinePlot),
    Scatter(ScatterPlot),
}

#[derive(Clone, Copy, Debug)]
enum HandleKind {
    Line,
    Scatter,
}

impl Default for BoxplotOptions {
    fn default() -> Self {
        Self {
            whisker: 1.5,
            orientation: Orientation::Vertical,
            labels: None,
            group_order: None,
            positions: None,
            widths: vec![0.5],
            outlier_marker: Some(MarkerStyle::Plus),
            outlier_color: Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
            outlier_size: 6.0,
            box_colors: vec![Vec4::new(0.0, 0.4470, 0.7410, 1.0)],
            color_group: None,
            data_lim: None,
            extreme_mode: ExtremeMode::Clip,
            jitter: 0.0,
            compact: false,
        }
    }
}

fn boxplot_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Unknown
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

fn value_is_integer_numeric(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn value_is_logical_numeric(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    }
}

fn value_contains_integer(value: &Value) -> bool {
    value_is_integer_numeric(value)
        || matches!(value, Value::Cell(cell) if cell.data.iter().any(value_contains_integer))
}

fn value_contains_logical(value: &Value) -> bool {
    value_is_logical_numeric(value)
        || matches!(value, Value::Cell(cell) if cell.data.iter().any(value_contains_logical))
}

fn value_contains_gpu(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

fn enable_boxplot_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)
}

fn ensure_boxplot_extensions_enabled(args: &[Value]) -> BuiltinResult<()> {
    let Some(x) = args.first() else {
        return Ok(());
    };
    if value_is_integer_numeric(x) {
        enable_boxplot_extension(&BOXPLOT_INTEGER_DATA_EXTENSION)?;
    }
    if value_is_logical_numeric(x) {
        enable_boxplot_extension(&BOXPLOT_LOGICAL_DATA_EXTENSION)?;
    }

    let mut idx = 1usize;
    if args.get(idx).is_some_and(|value| !is_option_name(value)) {
        let group = &args[idx];
        if value_contains_integer(group) {
            enable_boxplot_extension(&BOXPLOT_INTEGER_GROUP_EXTENSION)?;
        }
        if value_contains_logical(group) {
            enable_boxplot_extension(&BOXPLOT_LOGICAL_GROUP_EXTENSION)?;
        }
        idx += 1;
    }
    while idx + 1 < args.len() {
        let name = keyword_of(&args[idx]);
        let value = &args[idx + 1];
        if name.as_deref() == Some("colorgroup") {
            if value_contains_integer(value) {
                enable_boxplot_extension(&BOXPLOT_INTEGER_GROUP_EXTENSION)?;
            }
            if value_contains_logical(value) {
                enable_boxplot_extension(&BOXPLOT_LOGICAL_GROUP_EXTENSION)?;
            }
        } else {
            if value_is_integer_numeric(value) {
                enable_boxplot_extension(&BOXPLOT_INTEGER_OPTION_EXTENSION)?;
            }
            if value_is_logical_numeric(value) {
                enable_boxplot_extension(&BOXPLOT_LOGICAL_OPTION_EXTENSION)?;
            }
        }
        idx += 2;
    }
    if args.iter().any(value_contains_gpu) {
        enable_boxplot_extension(&BOXPLOT_GPU_INPUT_EXTENSION)?;
    }
    Ok(())
}

#[runtime_builtin(
    name = "boxplot",
    category = "stats/summary",
    summary = "Visualize sample quartiles, whiskers, and outliers with box plots.",
    keywords = "boxplot,box plot,statistics,summary,quartile,outlier,visualization",
    sink = true,
    suppress_auto_output = true,
    type_resolver(boxplot_type),
    descriptor(crate::builtins::stats::summary::boxplot::BOXPLOT_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::boxplot::BOXPLOT_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::boxplot::BOXPLOT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::boxplot"
)]
pub(crate) async fn boxplot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (axes_target, args) = split_leading_axes_handle(args, NAME).map_err(map_plot_error)?;
    ensure_boxplot_extensions_enabled(&args)?;
    let mut args = gather_values(args).await?;
    if args.is_empty() {
        return Err(invalid("boxplot: expected numeric data"));
    }
    let x = args.remove(0);
    let group = if args.first().is_some_and(|value| !is_option_name(value)) {
        Some(args.remove(0))
    } else {
        None
    };
    let options = BoxplotOptions::parse(args)?;
    let boxes = build_box_stats(x, group, &options)?;

    apply_axes_target(axes_target, NAME).map_err(map_plot_error)?;
    let handles = render_boxplot(&boxes, &options)?;
    let value = handles_value(handles)?;

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![value],
        )),
        None => Ok(value),
    }
}

impl BoxplotOptions {
    fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        let mut options = Self::default();
        if !args.len().is_multiple_of(2) {
            return Err(invalid("boxplot: name-value options must be paired"));
        }
        let mut idx = 0;
        while idx < args.len() {
            let name = keyword_of(&args[idx])
                .ok_or_else(|| invalid("boxplot: option names must be text"))?;
            let value = &args[idx + 1];
            match name.as_str() {
                "whisker" => {
                    let whisker = scalar(value, "Whisker")?;
                    if !whisker.is_finite() || whisker < 0.0 {
                        return Err(invalid(
                            "boxplot: Whisker must be a nonnegative finite scalar",
                        ));
                    }
                    options.whisker = whisker;
                }
                "orientation" => {
                    options.orientation = match text_scalar(value, "Orientation")?.as_str() {
                        "vertical" => Orientation::Vertical,
                        "horizontal" => Orientation::Horizontal,
                        _ => {
                            return Err(invalid(
                                "boxplot: Orientation must be 'vertical' or 'horizontal'",
                            ));
                        }
                    };
                }
                "labels" => {
                    options.labels = Some(text_vector(value, "Labels")?);
                }
                "grouporder" => {
                    options.group_order = Some(text_vector(value, "GroupOrder")?);
                }
                "positions" => {
                    let positions = numeric_vector(value, "Positions")?;
                    if positions.iter().any(|v| !v.is_finite()) {
                        return Err(invalid("boxplot: Positions must be finite"));
                    }
                    options.positions = Some(positions);
                }
                "widths" => {
                    let widths = numeric_vector(value, "Widths")?;
                    if widths.is_empty() || widths.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                        return Err(invalid("boxplot: Widths must be positive finite values"));
                    }
                    options.widths = widths;
                }
                "outliersize" => {
                    let size = scalar(value, "OutlierSize")?;
                    if !size.is_finite() || size <= 0.0 {
                        return Err(invalid(
                            "boxplot: OutlierSize must be a positive finite scalar",
                        ));
                    }
                    options.outlier_size = size as f32;
                }
                "symbol" => {
                    let text = value_text_preserve_case(value, "Symbol")?;
                    let (marker, color) = parse_symbol(&text)?;
                    options.outlier_marker = marker;
                    options.outlier_color = color;
                }
                "colors" => {
                    options.box_colors = parse_colors(value)?;
                }
                "plotstyle" => match text_scalar(value, "PlotStyle")?.as_str() {
                    "traditional" => {
                        options.compact = false;
                        if options.outlier_marker == Some(MarkerStyle::Circle) {
                            options.outlier_marker = Some(MarkerStyle::Plus);
                        }
                        options.outlier_size = 6.0;
                        options.jitter = 0.0;
                    }
                    "compact" => {
                        options.compact = true;
                        options.widths = vec![0.25];
                        options.outlier_marker = Some(MarkerStyle::Circle);
                        options.outlier_color = None;
                        options.outlier_size = 4.0;
                        options.jitter = 0.5;
                    }
                    _ => {
                        return Err(invalid(
                            "boxplot: PlotStyle must be 'traditional' or 'compact'",
                        ));
                    }
                },
                "boxstyle" => {
                    let text = text_scalar(value, "BoxStyle")?;
                    if text != "outline" && text != "filled" {
                        return Err(invalid("boxplot: BoxStyle must be 'outline' or 'filled'"));
                    }
                }
                "medianstyle" => {
                    let text = text_scalar(value, "MedianStyle")?;
                    if text != "line" && text != "target" {
                        return Err(invalid("boxplot: MedianStyle must be 'line' or 'target'"));
                    }
                }
                "notch" => {
                    parse_on_off_marker(value, "Notch")?;
                }
                "jitter" => {
                    let jitter = scalar(value, "Jitter")?;
                    if !jitter.is_finite() || jitter < 0.0 {
                        return Err(invalid("boxplot: Jitter must be nonnegative and finite"));
                    }
                    options.jitter = jitter;
                }
                "datalim" => {
                    let lim = numeric_vector(value, "DataLim")?;
                    if lim.len() != 2 || lim.iter().any(|v| v.is_nan()) || lim[0] > lim[1] {
                        return Err(invalid(
                            "boxplot: DataLim must be a two-element numeric range",
                        ));
                    }
                    options.data_lim = finite_data_lim(lim[0], lim[1]);
                }
                "extrememode" => {
                    options.extreme_mode = match text_scalar(value, "ExtremeMode")?.as_str() {
                        "clip" => ExtremeMode::Clip,
                        "compress" => ExtremeMode::Compress,
                        _ => {
                            return Err(invalid(
                                "boxplot: ExtremeMode must be 'compress' or 'clip'",
                            ));
                        }
                    };
                }
                "labelorientation" => {
                    let text = text_scalar(value, "LabelOrientation")?;
                    if text != "inline" && text != "horizontal" {
                        return Err(invalid(
                            "boxplot: LabelOrientation must be 'inline' or 'horizontal'",
                        ));
                    }
                }
                "labelverbosity" => {
                    let text = text_scalar(value, "LabelVerbosity")?;
                    if text != "all" && text != "minor" && text != "majorminor" {
                        return Err(invalid(
                            "boxplot: LabelVerbosity must be 'all', 'minor', or 'majorminor'",
                        ));
                    }
                }
                "colorgroup" => {
                    options.color_group = Some(value.clone());
                }
                "factordirection" => {
                    validate_choice_vector(value, "FactorDirection", &["data", "list"])?;
                }
                "fullfactors" => {
                    parse_on_off(value, "FullFactors")?;
                }
                "factorgap" => {
                    let gaps = numeric_vector(value, "FactorGap")?;
                    if gaps.iter().any(|v| !v.is_finite() || *v < 0.0) {
                        return Err(invalid(
                            "boxplot: FactorGap must contain nonnegative finite values",
                        ));
                    }
                }
                "factorseparator" => {
                    validate_factor_separator(value)?;
                }
                other => {
                    return Err(invalid(format!("boxplot: unsupported option '{other}'")));
                }
            }
            idx += 2;
        }
        Ok(options)
    }
}

fn is_option_name(value: &Value) -> bool {
    matches!(
        keyword_of(value).as_deref(),
        Some(
            "boxstyle"
                | "colors"
                | "medianstyle"
                | "notch"
                | "outliersize"
                | "plotstyle"
                | "symbol"
                | "widths"
                | "colorgroup"
                | "factordirection"
                | "fullfactors"
                | "factorgap"
                | "factorseparator"
                | "grouporder"
                | "datalim"
                | "extrememode"
                | "jitter"
                | "whisker"
                | "labels"
                | "labelorientation"
                | "labelverbosity"
                | "orientation"
                | "positions"
        )
    )
}

async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gather_if_needed_async(&arg).await?);
    }
    Ok(gathered)
}

fn build_box_stats(
    x: Value,
    group: Option<Value>,
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<BoxStats>> {
    let tensor =
        tensor::value_into_tensor_for(NAME, x).map_err(|err| invalid(format!("{NAME}: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(invalid("boxplot: X must be a vector or matrix"));
    }
    let tensor_len = tensor::tensor_element_len(&tensor);
    if tensor_len == 0 {
        return Err(invalid("boxplot: X must not be empty"));
    }
    let mut groups = match group {
        Some(group) => grouped_series(&tensor, group, options)?,
        None => ungrouped_series(&tensor),
    };
    if let Some(color_group) = &options.color_group {
        validate_group_like(&tensor, color_group)?;
    }
    groups.retain(|(_, values)| values.iter().any(|value| !value.is_nan()));
    if groups.is_empty() {
        return Err(invalid(
            "boxplot: data must contain at least one non-NaN value",
        ));
    }
    apply_group_order(&mut groups, options)?;
    if let Some(labels) = &options.labels {
        if labels.len() != groups.len() {
            return Err(invalid(
                "boxplot: Labels length must match the number of plotted boxes",
            ));
        }
        for (idx, (label, _)) in groups.iter_mut().enumerate() {
            *label = labels[idx].clone();
        }
    }
    let positions = positions_for(groups.len(), options)?;
    let widths = widths_for(groups.len(), options);
    groups
        .into_iter()
        .enumerate()
        .map(|(idx, (label, values))| {
            let mut stats = stats_for_values(label, values, options.whisker)?;
            stats.position = positions[idx];
            stats.width = widths[idx];
            stats.color = color_for(idx, &options.box_colors);
            Ok(stats)
        })
        .collect()
}

fn ungrouped_series(tensor: &Tensor) -> Vec<(String, Vec<f64>)> {
    let values = tensor::tensor_values_f64_cow(tensor);
    if tensor.rows == 1 || tensor.cols == 1 {
        return vec![("1".to_string(), values.to_vec())];
    }
    (0..tensor.cols)
        .map(|col| {
            let values = (0..tensor.rows)
                .map(|row| values[col * tensor.rows + row])
                .collect::<Vec<_>>();
            ((col + 1).to_string(), values)
        })
        .collect()
}

fn grouped_series(
    tensor: &Tensor,
    group: Value,
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<(String, Vec<f64>)>> {
    let values = tensor::tensor_values_f64_cow(tensor);
    let tensor_len = values.len();
    let labels = match group_labels(&group, tensor_len) {
        Ok(labels) => labels,
        Err(err) if tensor.rows > 1 && tensor.cols > 1 => {
            let row_labels = group_labels(&group, tensor.rows).map_err(|_| err)?;
            let mut labels = Vec::with_capacity(tensor_len);
            for _col in 0..tensor.cols {
                labels.extend(row_labels.iter().cloned());
            }
            labels
        }
        Err(err) => return Err(err),
    };
    if labels.len() != tensor_len {
        return Err(invalid(
            "boxplot: grouping variables must have the same number of elements as X",
        ));
    }
    let mut first_seen = Vec::<String>::new();
    let mut text_groups = HashMap::<String, Vec<f64>>::new();
    let mut floating_groups = BTreeMap::<OrderedF64, (String, Vec<f64>)>::new();
    let mut integer_groups = BTreeMap::<i128, (String, Vec<f64>)>::new();
    let floating_order = labels
        .iter()
        .flatten()
        .all(|label| matches!(label, GroupLabel::Floating(_)));
    let integer_order = labels
        .iter()
        .flatten()
        .all(|label| matches!(label, GroupLabel::Integer(_)));

    for (idx, maybe_label) in labels.into_iter().enumerate() {
        let Some(label) = maybe_label else {
            continue;
        };
        if floating_order {
            let GroupLabel::Floating(value) = label else {
                unreachable!("floating group classification checked above");
            };
            let text = number_label(&value);
            floating_groups
                .entry(OrderedF64(value))
                .or_insert_with(|| (text, Vec::new()))
                .1
                .push(values[idx]);
        } else if integer_order {
            let GroupLabel::Integer(value) = label else {
                unreachable!("integer group classification checked above");
            };
            integer_groups
                .entry(value)
                .or_insert_with(|| (value.to_string(), Vec::new()))
                .1
                .push(values[idx]);
        } else {
            let text = label.text();
            if !text_groups.contains_key(&text) {
                first_seen.push(text.clone());
            }
            text_groups.entry(text).or_default().push(values[idx]);
        }
    }

    let groups = if floating_order {
        floating_groups.into_values().collect::<Vec<_>>()
    } else if integer_order {
        integer_groups.into_values().collect::<Vec<_>>()
    } else {
        first_seen
            .into_iter()
            .filter_map(|label| text_groups.remove(&label).map(|values| (label, values)))
            .collect::<Vec<_>>()
    };

    if groups.is_empty() && options.group_order.is_some() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(groups)
}

fn apply_group_order(
    groups: &mut Vec<(String, Vec<f64>)>,
    options: &BoxplotOptions,
) -> BuiltinResult<()> {
    let Some(order) = &options.group_order else {
        return Ok(());
    };
    let mut by_label = groups.drain(..).collect::<HashMap<_, _>>();
    for label in order {
        if let Some(values) = by_label.remove(label) {
            groups.push((label.clone(), values));
        }
    }
    if groups.is_empty() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(())
}

fn stats_for_values(label: String, values: Vec<f64>, whisker: f64) -> BuiltinResult<BoxStats> {
    let mut finite = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err(invalid(
            "boxplot: each plotted box must contain at least one non-NaN value",
        ));
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let q1 = percentile_sorted(&finite, 0.25);
    let median = percentile_sorted(&finite, 0.5);
    let q3 = percentile_sorted(&finite, 0.75);
    let iqr = q3 - q1;
    let lower_fence = q1 - whisker * iqr;
    let upper_fence = q3 + whisker * iqr;
    let mut lower_whisker = None;
    let mut upper_whisker = None;
    let mut outliers = Vec::new();
    for value in finite {
        if value < lower_fence || value > upper_fence {
            outliers.push(value);
        } else {
            lower_whisker = Some(lower_whisker.map_or(value, |old: f64| old.min(value)));
            upper_whisker = Some(upper_whisker.map_or(value, |old: f64| old.max(value)));
        }
    }
    Ok(BoxStats {
        label,
        position: 0.0,
        width: 0.5,
        color: Vec4::ONE,
        q1,
        median,
        q3,
        lower_whisker: lower_whisker.unwrap_or(median),
        upper_whisker: upper_whisker.unwrap_or(median),
        outliers,
    })
}

fn percentile_sorted(values: &[f64], p: f64) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    let pos = p * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let frac = pos - lo as f64;
        values[lo] * (1.0 - frac) + values[hi] * frac
    }
}

fn render_boxplot(boxes: &[BoxStats], options: &BoxplotOptions) -> BuiltinResult<Vec<f64>> {
    let plans = build_plot_plans(boxes, options)?;
    let labels = boxes
        .iter()
        .map(|stats| stats.label.clone())
        .collect::<Vec<_>>();
    let limits = axis_limits(boxes, options);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let mut plans = Some(plans);
    let handles_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let handles_slot = std::rc::Rc::clone(&handles_out);
    let opts = PlotRenderOptions {
        title: "",
        x_label: "",
        y_label: "",
        grid: true,
        axis_equal: false,
    };
    let render_result = render_active_plot(NAME, opts, move |figure, axes| {
        match options.orientation {
            Orientation::Vertical => {
                figure.set_axes_tick_labels(axes, Some(labels.clone()), None);
            }
            Orientation::Horizontal => {
                figure.set_axes_tick_labels(axes, None, Some(labels.clone()));
            }
        }
        figure.set_axes_limits(axes, limits.0, limits.1);
        let mut recorded = Vec::new();
        for plan in plans.take().expect("boxplot plans consumed once") {
            match plan {
                PlotPlan::Line(plot) => {
                    let plot_index = figure.add_line_plot_on_axes(plot, axes);
                    recorded.push((HandleKind::Line, axes, plot_index));
                }
                PlotPlan::Scatter(plot) => {
                    let plot_index = figure.add_scatter_plot_on_axes(plot, axes);
                    recorded.push((HandleKind::Scatter, axes, plot_index));
                }
            }
        }
        *handles_slot.borrow_mut() = recorded;
        Ok(())
    });

    let handles = handles_out
        .borrow()
        .iter()
        .map(|(kind, axes, plot_index)| match kind {
            HandleKind::Line => crate::builtins::plotting::state::register_line_handle(
                figure_handle,
                *axes,
                *plot_index,
            ),
            HandleKind::Scatter => crate::builtins::plotting::state::register_scatter_handle(
                figure_handle,
                *axes,
                *plot_index,
            ),
        })
        .collect::<Vec<_>>();

    if let Err(err) = render_result {
        let lower = err.to_string().to_ascii_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handles);
        }
        return Err(internal(err.message));
    }
    Ok(handles)
}

type AxisLimits = (Option<(f64, f64)>, Option<(f64, f64)>);

fn axis_limits(boxes: &[BoxStats], options: &BoxplotOptions) -> AxisLimits {
    let min_pos = boxes
        .iter()
        .map(|stats| stats.position - stats.width)
        .fold(f64::INFINITY, f64::min);
    let max_pos = boxes
        .iter()
        .map(|stats| stats.position + stats.width)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut min_data = f64::INFINITY;
    let mut max_data = f64::NEG_INFINITY;
    for stats in boxes {
        min_data = min_data.min(stats.lower_whisker).min(stats.q1);
        max_data = max_data.max(stats.upper_whisker).max(stats.q3);
        for (idx, value) in stats.outliers.iter().enumerate() {
            let value = display_data_value(*value, idx, stats.outliers.len(), options);
            min_data = min_data.min(value);
            max_data = max_data.max(value);
        }
    }
    if let Some((lo, hi)) = options.data_lim {
        if lo.is_finite() {
            min_data = lo.min(min_data);
        }
        if hi.is_finite() {
            max_data = hi.max(max_data);
        }
    }
    if !min_data.is_finite() || !max_data.is_finite() {
        min_data = 0.0;
        max_data = 1.0;
    }
    if (max_data - min_data).abs() < f64::EPSILON {
        min_data -= 0.5;
        max_data += 0.5;
    } else {
        let pad = (max_data - min_data).abs() * 0.08;
        min_data -= pad;
        max_data += pad;
    }
    let category = Some((min_pos.min(0.5), max_pos.max(boxes.len() as f64 + 0.5)));
    let data = Some((min_data, max_data));
    match options.orientation {
        Orientation::Vertical => (category, data),
        Orientation::Horizontal => (data, category),
    }
}

fn build_plot_plans(boxes: &[BoxStats], options: &BoxplotOptions) -> BuiltinResult<Vec<PlotPlan>> {
    let mut plans = Vec::new();
    for stats in boxes {
        add_box_lines(&mut plans, stats, options.orientation)?;
        if !stats.outliers.is_empty() {
            if let Some(marker) = options.outlier_marker {
                let (x, y) = outlier_points(stats, options);
                let color = options.outlier_color.unwrap_or(stats.color);
                let mut scatter = ScatterPlot::new(x, y)
                    .map_err(|err| internal(format!("boxplot: {err}")))?
                    .with_style(color, options.outlier_size, marker)
                    .with_label(format!("{} outliers", stats.label));
                scatter.set_edge_color(color);
                scatter.set_filled(false);
                plans.push(PlotPlan::Scatter(scatter));
            }
        }
    }
    Ok(plans)
}

fn add_box_lines(
    plans: &mut Vec<PlotPlan>,
    stats: &BoxStats,
    orientation: Orientation,
) -> BuiltinResult<()> {
    let pos = stats.position;
    let half = stats.width * 0.5;
    let cap_half = stats.width * 0.25;
    let left = pos - half;
    let right = pos + half;
    let cap_left = pos - cap_half;
    let cap_right = pos + cap_half;
    let segments = [
        (
            vec![left, right, right, left, left],
            vec![stats.q1, stats.q1, stats.q3, stats.q3, stats.q1],
            "box",
        ),
        (
            vec![left, right],
            vec![stats.median, stats.median],
            "median",
        ),
        (
            vec![pos, pos],
            vec![stats.q1, stats.lower_whisker],
            "lower whisker",
        ),
        (
            vec![pos, pos],
            vec![stats.q3, stats.upper_whisker],
            "upper whisker",
        ),
        (
            vec![cap_left, cap_right],
            vec![stats.lower_whisker, stats.lower_whisker],
            "lower cap",
        ),
        (
            vec![cap_left, cap_right],
            vec![stats.upper_whisker, stats.upper_whisker],
            "upper cap",
        ),
    ];
    for (x, y, suffix) in segments {
        let (x, y) = orient_points(x, y, orientation);
        let line = LinePlot::new(x, y)
            .map_err(|err| internal(format!("boxplot: {err}")))?
            .with_style(stats.color, 1.5, LineStyle::Solid)
            .with_label(format!("{} {}", stats.label, suffix));
        plans.push(PlotPlan::Line(line));
    }
    Ok(())
}

fn outlier_points(stats: &BoxStats, options: &BoxplotOptions) -> (Vec<f64>, Vec<f64>) {
    let mut category = Vec::with_capacity(stats.outliers.len());
    for idx in 0..stats.outliers.len() {
        let offset = if options.jitter == 0.0 || stats.outliers.len() == 1 {
            0.0
        } else {
            ((idx % 5) as f64 - 2.0) * stats.width * options.jitter * 0.07
        };
        category.push(stats.position + offset);
    }
    let values = stats
        .outliers
        .iter()
        .enumerate()
        .map(|(idx, value)| display_data_value(*value, idx, stats.outliers.len(), options))
        .collect();
    orient_points(category, values, options.orientation)
}

fn orient_points(x: Vec<f64>, y: Vec<f64>, orientation: Orientation) -> (Vec<f64>, Vec<f64>) {
    match orientation {
        Orientation::Vertical => (x, y),
        Orientation::Horizontal => (y, x),
    }
}

fn handles_value(handles: Vec<f64>) -> BuiltinResult<Value> {
    let len = handles.len();
    Tensor::new(handles, vec![1, len])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("boxplot: {err}")))
}

fn positions_for(count: usize, options: &BoxplotOptions) -> BuiltinResult<Vec<f64>> {
    if let Some(positions) = &options.positions {
        if positions.len() != count {
            return Err(invalid(
                "boxplot: Positions length must match the number of plotted boxes",
            ));
        }
        return Ok(positions.clone());
    }
    Ok((1..=count).map(|idx| idx as f64).collect())
}

fn widths_for(count: usize, options: &BoxplotOptions) -> Vec<f64> {
    (0..count)
        .map(|idx| options.widths[idx % options.widths.len()])
        .collect()
}

fn color_for(idx: usize, colors: &[Vec4]) -> Vec4 {
    colors
        .get(idx % colors.len().max(1))
        .copied()
        .unwrap_or(Vec4::new(0.0, 0.4470, 0.7410, 1.0))
}

fn numeric_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(NAME, value.clone())
        .map_err(|err| invalid(format!("boxplot: {label} must be numeric ({err})")))?;
    Ok(tensor::tensor_into_values_f64(tensor))
}

fn scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => {
            let values = numeric_vector(other, label)?;
            if values.len() == 1 {
                Ok(values[0])
            } else {
                Err(invalid(format!("boxplot: {label} must be scalar")))
            }
        }
    }
}

fn text_scalar(value: &Value, label: &str) -> BuiltinResult<String> {
    value_text_preserve_case(value, label).map(|text| text.to_ascii_lowercase())
}

fn value_text_preserve_case(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        _ => Err(invalid(format!("boxplot: {label} must be text"))),
    }
}

fn text_vector(value: &Value, label: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) => Ok(char_rows(array)),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| value_text_preserve_case(value, label))
            .collect(),
        Value::Tensor(tensor) => (0..tensor::tensor_element_len(tensor))
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .map(numeric_scalar_label)
                    .ok_or_else(|| invalid(format!("boxplot: {label} storage is malformed")))
            })
            .collect(),
        Value::Num(n) => Ok(vec![number_label(n)]),
        Value::Int(i) => Ok(vec![int_group_key(i).to_string()]),
        _ => Err(invalid(format!("boxplot: {label} must be a text vector"))),
    }
}

fn parse_on_off_marker(value: &Value, label: &str) -> BuiltinResult<()> {
    match text_scalar(value, label)?.as_str() {
        "on" | "off" | "marker" => Ok(()),
        _ => Err(invalid(format!(
            "boxplot: {label} must be 'on', 'off', or 'marker'"
        ))),
    }
}

fn parse_on_off(value: &Value, label: &str) -> BuiltinResult<()> {
    match text_scalar(value, label)?.as_str() {
        "on" | "off" => Ok(()),
        _ => Err(invalid(format!("boxplot: {label} must be 'on' or 'off'"))),
    }
}

fn validate_choice_vector(value: &Value, label: &str, choices: &[&str]) -> BuiltinResult<()> {
    let values = text_vector(value, label)?;
    if values.is_empty() {
        return Err(invalid(format!("boxplot: {label} must not be empty")));
    }
    for value in values {
        let value = value.to_ascii_lowercase();
        if !choices.iter().any(|choice| value == *choice) {
            return Err(invalid(format!(
                "boxplot: {label} must be one of {}",
                choices.join(", ")
            )));
        }
    }
    Ok(())
}

fn validate_factor_separator(value: &Value) -> BuiltinResult<()> {
    if let Some(text) = keyword_of(value) {
        if text == "auto" {
            return Ok(());
        }
        return Err(invalid(
            "boxplot: FactorSeparator must be numeric or 'auto'",
        ));
    }
    let values = numeric_vector(value, "FactorSeparator")?;
    if values.iter().any(|value| {
        !value.is_finite() || *value < 1.0 || (*value - value.round()).abs() > f64::EPSILON
    }) {
        return Err(invalid(
            "boxplot: FactorSeparator must contain positive integer indices",
        ));
    }
    Ok(())
}

fn finite_data_lim(lo: f64, hi: f64) -> Option<(f64, f64)> {
    if lo == f64::NEG_INFINITY && hi == f64::INFINITY {
        None
    } else {
        Some((lo, hi))
    }
}

fn display_data_value(value: f64, idx: usize, count: usize, options: &BoxplotOptions) -> f64 {
    let Some((lo, hi)) = options.data_lim else {
        return value;
    };
    if value < lo && lo.is_finite() {
        return match options.extreme_mode {
            ExtremeMode::Clip => lo,
            ExtremeMode::Compress => lo - compression_offset(lo, hi, idx, count),
        };
    }
    if value > hi && hi.is_finite() {
        return match options.extreme_mode {
            ExtremeMode::Clip => hi,
            ExtremeMode::Compress => hi + compression_offset(lo, hi, idx, count),
        };
    }
    value
}

fn compression_offset(lo: f64, hi: f64, idx: usize, count: usize) -> f64 {
    let span = if lo.is_finite() && hi.is_finite() && hi > lo {
        hi - lo
    } else {
        1.0
    };
    let stack = if count <= 1 { 0.0 } else { (idx % 3) as f64 };
    span.max(1.0) * (0.03 + stack * 0.015)
}

fn parse_symbol(text: &str) -> BuiltinResult<(Option<MarkerStyle>, Option<Vec4>)> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return Ok((None, None));
    }
    let lower = trimmed.to_ascii_lowercase();
    let named = match lower.as_str() {
        "plus" => Some(MarkerStyle::Plus),
        "circle" | "o" => Some(MarkerStyle::Circle),
        "point" | "dot" => Some(MarkerStyle::Circle),
        "cross" | "x" => Some(MarkerStyle::Cross),
        "star" | "asterisk" => Some(MarkerStyle::Star),
        "square" => Some(MarkerStyle::Square),
        "diamond" => Some(MarkerStyle::Diamond),
        "triangle" | "uptriangle" | "downtriangle" | "righttriangle" | "lefttriangle" => {
            Some(MarkerStyle::Triangle)
        }
        "vline" | "vertical" => Some(MarkerStyle::Plus),
        "hline" | "horizontal" => Some(MarkerStyle::Square),
        _ => None,
    };
    if let Some(marker) = named {
        return Ok((Some(marker), None));
    }
    let mut marker = None;
    let mut color = None;
    for ch in trimmed.chars() {
        match ch {
            '+' => marker = Some(MarkerStyle::Plus),
            'o' | 'O' => marker = Some(MarkerStyle::Circle),
            '.' => marker = Some(MarkerStyle::Circle),
            'x' | 'X' => marker = Some(MarkerStyle::Cross),
            '*' => marker = Some(MarkerStyle::Star),
            's' | 'S' => marker = Some(MarkerStyle::Square),
            'd' | 'D' => marker = Some(MarkerStyle::Diamond),
            '^' | 'v' | '>' | '<' => marker = Some(MarkerStyle::Triangle),
            '|' => marker = Some(MarkerStyle::Plus),
            '_' => marker = Some(MarkerStyle::Square),
            'y' | 'm' | 'c' | 'r' | 'g' | 'b' | 'w' | 'k' => {
                color = Some(color_char(ch).expect("known color"))
            }
            ' ' => {}
            other => {
                return Err(invalid(format!(
                    "boxplot: unsupported Symbol character '{other}'"
                )));
            }
        }
    }
    Ok((marker, color))
}

fn parse_colors(value: &Value) -> BuiltinResult<Vec<Vec4>> {
    if let Some(text) = match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    } {
        let colors = text
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .map(|ch| {
                color_char(ch)
                    .ok_or_else(|| invalid(format!("boxplot: unsupported Colors character '{ch}'")))
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        if colors.is_empty() {
            return Err(invalid("boxplot: Colors must not be empty"));
        }
        return Ok(colors);
    }
    let tensor = tensor::value_into_tensor_for(NAME, value.clone()).map_err(|err| {
        invalid(format!(
            "boxplot: Colors must be text or RGB matrix ({err})"
        ))
    })?;
    if tensor.cols != 3 || tensor.rows == 0 {
        return Err(invalid(
            "boxplot: Colors RGB matrix must have three columns",
        ));
    }
    let values = tensor::tensor_values_f64_cow(&tensor);
    let mut colors = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let r = values[row];
        let g = values[tensor.rows + row];
        let b = values[2 * tensor.rows + row];
        if !(0.0..=1.0).contains(&r) || !(0.0..=1.0).contains(&g) || !(0.0..=1.0).contains(&b) {
            return Err(invalid("boxplot: RGB values must be in the range [0,1]"));
        }
        colors.push(Vec4::new(r as f32, g as f32, b as f32, 1.0));
    }
    Ok(colors)
}

fn color_char(ch: char) -> Option<Vec4> {
    match ch.to_ascii_lowercase() {
        'y' => Some(Vec4::new(1.0, 1.0, 0.0, 1.0)),
        'm' => Some(Vec4::new(1.0, 0.0, 1.0, 1.0)),
        'c' => Some(Vec4::new(0.0, 1.0, 1.0, 1.0)),
        'r' => Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        'g' => Some(Vec4::new(0.0, 0.5, 0.0, 1.0)),
        'b' => Some(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        'w' => Some(Vec4::new(1.0, 1.0, 1.0, 1.0)),
        'k' => Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        _ => None,
    }
}

#[derive(Clone, Debug)]
enum GroupLabel {
    Floating(f64),
    Integer(i128),
    Text(String),
}

impl GroupLabel {
    fn text(&self) -> String {
        match self {
            Self::Floating(value) => number_label(value),
            Self::Integer(value) => value.to_string(),
            Self::Text(value) => value.clone(),
        }
    }
}

fn int_group_key(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn numeric_group_label(value: NumericScalar) -> Option<GroupLabel> {
    match value {
        NumericScalar::F64(value) => value.is_finite().then_some(GroupLabel::Floating(value)),
        NumericScalar::F32(value) => value
            .is_finite()
            .then_some(GroupLabel::Floating(f64::from(value))),
        NumericScalar::I8(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I16(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I32(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I64(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U8(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U16(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U32(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U64(value) => Some(GroupLabel::Integer(i128::from(value))),
    }
}

fn group_labels(value: &Value, expected_len: usize) -> BuiltinResult<Vec<Option<GroupLabel>>> {
    match value {
        Value::Cell(cell) => group_labels_from_cell(cell, expected_len),
        Value::Object(object) if object.is_class("categorical") => (0..expected_len)
            .map(|row| {
                let label = crate::builtins::table::categorical_label_at(object, row)
                    .ok_or_else(|| invalid("boxplot: invalid categorical grouping variable"))?;
                Ok(missing_text_label(&label).then_some(GroupLabel::Text(label)))
            })
            .collect(),
        Value::String(text) if expected_len == 1 => Ok(vec![Some(GroupLabel::Text(text.clone()))]),
        Value::StringArray(array) if array.data.len() == expected_len => Ok(array
            .data
            .iter()
            .map(|text| missing_text_label(text).then(|| GroupLabel::Text(text.clone())))
            .collect()),
        Value::CharArray(array) if array.rows == expected_len => Ok(char_rows(array)
            .into_iter()
            .map(|text| missing_text_label(&text).then_some(GroupLabel::Text(text)))
            .collect()),
        Value::CharArray(array) if array.rows == 1 && expected_len == 1 => {
            let text: String = array.data.iter().collect();
            Ok(vec![
                missing_text_label(&text).then_some(GroupLabel::Text(text))
            ])
        }
        _ => {
            let tensor = tensor::value_into_tensor_for(NAME, value.clone()).map_err(|err| {
                invalid(format!(
                    "boxplot: grouping variable must be numeric, text, cell, or categorical ({err})"
                ))
            })?;
            if tensor::tensor_element_len(&tensor) != expected_len {
                return Err(invalid(
                    "boxplot: grouping variable length must match the number of X elements",
                ));
            }
            Ok((0..expected_len)
                .map(|index| tensor.numeric_value_at(index).and_then(numeric_group_label))
                .collect())
        }
    }
}

fn validate_group_like(tensor: &Tensor, value: &Value) -> BuiltinResult<()> {
    match group_labels(value, tensor::tensor_element_len(tensor)) {
        Ok(_) => Ok(()),
        Err(err) if tensor.rows > 1 && tensor.cols > 1 => group_labels(value, tensor.rows)
            .map(|_| ())
            .map_err(|_| err),
        Err(err) => Err(err),
    }
}

fn group_labels_from_cell(
    cell: &CellArray,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<GroupLabel>>> {
    if cell.data.len() == expected_len {
        let scalar_labels = cell
            .data
            .iter()
            .map(scalar_group_label)
            .collect::<BuiltinResult<Vec<_>>>();
        if let Ok(labels) = scalar_labels {
            return Ok(labels);
        }
    }
    let mut variables = Vec::with_capacity(cell.data.len());
    for value in &cell.data {
        let labels = group_labels(value, expected_len)?;
        if labels.len() != expected_len {
            return Err(invalid(
                "boxplot: grouping variables in a cell array must share the X length",
            ));
        }
        variables.push(labels);
    }
    let mut labels = Vec::with_capacity(expected_len);
    for row in 0..expected_len {
        let mut parts = Vec::with_capacity(variables.len());
        let mut missing = false;
        for variable in &variables {
            match &variable[row] {
                Some(label) => parts.push(label.text()),
                None => {
                    missing = true;
                    break;
                }
            }
        }
        labels.push((!missing).then(|| GroupLabel::Text(parts.join(","))));
    }
    Ok(labels)
}

fn scalar_group_label(value: &Value) -> BuiltinResult<Option<GroupLabel>> {
    match value {
        Value::String(text) => Ok(missing_text_label(text).then(|| GroupLabel::Text(text.clone()))),
        Value::StringArray(array) if array.data.len() == 1 => {
            let text = array.data[0].clone();
            Ok(missing_text_label(&text).then_some(GroupLabel::Text(text)))
        }
        Value::CharArray(array) if array.rows == 1 => {
            let text: String = array.data.iter().collect();
            Ok(missing_text_label(&text).then_some(GroupLabel::Text(text)))
        }
        Value::Num(value) => Ok(value.is_finite().then_some(GroupLabel::Floating(*value))),
        Value::Int(value) => Ok(Some(GroupLabel::Integer(int_group_key(value)))),
        Value::Bool(value) => Ok(Some(GroupLabel::Text(
            if *value { "true" } else { "false" }.to_string(),
        ))),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor.numeric_value_at(0).and_then(numeric_group_label))
        }
        _ => Err(invalid("boxplot: cell grouping labels must be scalar")),
    }
}

fn missing_text_label(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && trimmed != "<missing>"
        && trimmed != "<undefined>"
        && !trimmed.eq_ignore_ascii_case("missing")
}

fn char_rows(array: &CharArray) -> Vec<String> {
    if array.rows <= 1 {
        return vec![array.data.iter().collect()];
    }
    (0..array.rows)
        .map(|row| {
            (0..array.cols)
                .filter_map(|col| array.data.get(row + col * array.rows).copied())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

fn number_label(value: &f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        let text = format!("{value:.15}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn numeric_scalar_label(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => number_label(&value),
        NumericScalar::F32(value) => number_label(&f64::from(value)),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::{clone_figure, current_figure_handle};
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{IntegerStorage, LogicalArray};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).expect("tensor"))
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn descriptor_declares_integer_forms_and_extensions() {
        let builtin = builtin_function_by_name("boxplot").expect("registered boxplot");
        assert_eq!(builtin.extensions, &BOXPLOT_EXTENSIONS);
        assert_eq!(
            builtin
                .integer_capabilities
                .iter()
                .map(|capability| capability.form)
                .collect::<Vec<_>>(),
            BOXPLOT_INTEGER_CAPABILITIES
                .iter()
                .map(|capability| capability.form)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn extensions_are_independently_mode_gated() {
        let data = || tensor(vec![1.0, 2.0], vec![2, 1]);
        let integer_data = || int_tensor(IntegerStorage::I16(vec![1, 2]), vec![2, 1]);
        let logical_data = || {
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical data"))
        };
        let integer_group = || int_tensor(IntegerStorage::U16(vec![1, 2]), vec![2, 1]);
        let logical_group = || {
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical group"))
        };
        let cases = [
            (
                vec![integer_data()],
                BOXPLOT_INTEGER_DATA_EXTENSION.error_identifier,
            ),
            (
                vec![logical_data()],
                BOXPLOT_LOGICAL_DATA_EXTENSION.error_identifier,
            ),
            (
                vec![data(), integer_group()],
                BOXPLOT_INTEGER_GROUP_EXTENSION.error_identifier,
            ),
            (
                vec![data(), logical_group()],
                BOXPLOT_LOGICAL_GROUP_EXTENSION.error_identifier,
            ),
            (
                vec![data(), Value::from("Whisker"), Value::Int(IntValue::U8(1))],
                BOXPLOT_INTEGER_OPTION_EXTENSION.error_identifier,
            ),
            (
                vec![data(), Value::from("Whisker"), Value::Bool(true)],
                BOXPLOT_LOGICAL_OPTION_EXTENSION.error_identifier,
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (args, identifier) in cases {
            let error =
                ensure_boxplot_extensions_enabled(&args).expect_err("strict mode must reject");
            assert_eq!(error.identifier(), identifier);
        }
    }

    #[test]
    fn documented_single_data_groups_and_options_need_no_extension() {
        let single = |values: Vec<f32>, shape: Vec<usize>| {
            Value::Tensor(Tensor::from_f32(values, shape).expect("single tensor"))
        };
        let args = vec![
            single(vec![1.0, 2.0], vec![2, 1]),
            single(vec![2.0, 1.0], vec![2, 1]),
            Value::from("Whisker"),
            single(vec![1.0], vec![1, 1]),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        ensure_boxplot_extensions_enabled(&args).expect("documented single forms");
    }

    #[test]
    fn all_integer_sample_classes_cross_the_explicit_statistics_boundary() {
        let cases = [
            (IntegerStorage::I8(vec![-1, 3]), 1.0),
            (IntegerStorage::I16(vec![-1, 3]), 1.0),
            (IntegerStorage::I32(vec![-1, 3]), 1.0),
            (IntegerStorage::I64(vec![-1, 3]), 1.0),
            (IntegerStorage::U8(vec![1, 3]), 2.0),
            (IntegerStorage::U16(vec![1, 3]), 2.0),
            (IntegerStorage::U32(vec![1, 3]), 2.0),
            (IntegerStorage::U64(vec![1, 3]), 2.0),
        ];
        for (storage, expected_median) in cases {
            let boxes = build_box_stats(
                int_tensor(storage, vec![2, 1]),
                None,
                &BoxplotOptions::default(),
            )
            .expect("integer box statistics");
            assert_eq!(boxes.len(), 1);
            assert_eq!(boxes[0].median, expected_median);
        }
    }

    #[test]
    fn all_integer_group_and_numeric_option_classes_are_covered() {
        let cases = [
            (IntegerStorage::I8(vec![2, 1]), IntValue::I8(1)),
            (IntegerStorage::I16(vec![2, 1]), IntValue::I16(1)),
            (IntegerStorage::I32(vec![2, 1]), IntValue::I32(1)),
            (IntegerStorage::I64(vec![2, 1]), IntValue::I64(1)),
            (IntegerStorage::U8(vec![2, 1]), IntValue::U8(1)),
            (IntegerStorage::U16(vec![2, 1]), IntValue::U16(1)),
            (IntegerStorage::U32(vec![2, 1]), IntValue::U32(1)),
            (IntegerStorage::U64(vec![2, 1]), IntValue::U64(1)),
        ];
        for (groups, option) in cases {
            let boxes = build_box_stats(
                tensor(vec![20.0, 10.0], vec![2, 1]),
                Some(int_tensor(groups, vec![2, 1])),
                &BoxplotOptions::default(),
            )
            .expect("integer groups");
            assert_eq!(
                boxes
                    .iter()
                    .map(|box_stats| box_stats.label.as_str())
                    .collect::<Vec<_>>(),
                vec!["1", "2"]
            );
            let options = BoxplotOptions::parse(vec![Value::from("Whisker"), Value::Int(option)])
                .expect("integer numeric option");
            assert_eq!(options.whisker, 1.0);
        }
    }

    #[test]
    fn integer_group_identity_and_labels_remain_exact_above_flintmax() {
        let wide = 9_007_199_254_740_993_u64;
        let boxes = build_box_stats(
            tensor(vec![10.0, 20.0, 30.0], vec![3, 1]),
            Some(int_tensor(
                IntegerStorage::U64(vec![wide + 1, wide, wide + 1]),
                vec![3, 1],
            )),
            &BoxplotOptions::default(),
        )
        .expect("wide integer grouping");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].label, wide.to_string());
        assert_eq!(boxes[0].median, 20.0);
        assert_eq!(boxes[1].label, (wide + 1).to_string());
        assert_eq!(boxes[1].median, 20.0);
    }

    #[test]
    fn integer_labels_preserve_exact_text() {
        let wide = 9_007_199_254_740_993_u64;
        assert_eq!(
            text_vector(
                &int_tensor(IntegerStorage::U64(vec![wide]), vec![1, 1]),
                "Labels",
            )
            .expect("integer labels"),
            vec![wide.to_string()]
        );
    }

    #[test]
    fn gpu_input_extension_rejects_before_gather_and_admits_host_rendering() {
        use runmat_accelerate_api::HostTensorView;

        crate::builtins::common::test_support::with_test_provider(|provider| {
            let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
            crate::builtins::plotting::reset_plot_state();
            let handle = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[2, 1],
                })
                .expect("resident data");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(boxplot_builtin(vec![Value::GpuTensor(handle.clone())]))
                    .expect_err("strict mode rejects resident input");
                assert_eq!(
                    error.identifier(),
                    BOXPLOT_GPU_INPUT_EXTENSION.error_identifier
                );
            }
            {
                let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
                let _outputs = crate::output_count::push_output_count(Some(0));
                assert_eq!(
                    block_on(boxplot_builtin(vec![Value::GpuTensor(handle)]))
                        .expect("RunMat resident input"),
                    Value::OutputList(Vec::new())
                );
            }
        });
    }

    #[test]
    fn computes_matrix_columns_and_outliers() {
        let options = BoxplotOptions::default();
        let boxes = build_box_stats(
            tensor(
                vec![1.0, 2.0, 3.0, 100.0, 10.0, 20.0, 30.0, 40.0],
                vec![4, 2],
            ),
            None,
            &options,
        )
        .expect("stats");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].label, "1");
        assert!((boxes[0].median - 2.5).abs() < 1.0e-12);
        assert!(boxes[0].outliers.contains(&100.0));
        assert_eq!(boxes[1].label, "2");
        assert!((boxes[1].q1 - 17.5).abs() < 1.0e-12);
    }

    #[test]
    fn groups_text_labels_with_group_order_and_labels() {
        let labels = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("b")),
                Value::CharArray(CharArray::new_row("a")),
                Value::CharArray(CharArray::new_row("b")),
                Value::CharArray(CharArray::new_row("a")),
            ],
            4,
            1,
        )
        .expect("cell");
        let options = BoxplotOptions {
            group_order: Some(vec!["a".to_string(), "b".to_string()]),
            labels: Some(vec!["A".to_string(), "B".to_string()]),
            ..Default::default()
        };
        let boxes = build_box_stats(
            tensor(vec![1.0, 10.0, 3.0, 20.0], vec![4, 1]),
            Some(Value::Cell(labels)),
            &options,
        )
        .expect("stats");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].label, "A");
        assert_eq!(boxes[1].label, "B");
        assert!((boxes[0].median - 15.0).abs() < 1.0e-12);
        assert!((boxes[1].median - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn matrix_grouping_accepts_row_group_vector() {
        let labels = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("low")),
                Value::CharArray(CharArray::new_row("high")),
            ],
            2,
            1,
        )
        .expect("cell");
        let boxes = build_box_stats(
            tensor(vec![1.0, 10.0, 2.0, 20.0], vec![2, 2]),
            Some(Value::Cell(labels)),
            &BoxplotOptions::default(),
        )
        .expect("stats");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].label, "low");
        assert_eq!(boxes[1].label, "high");
        assert!((boxes[0].median - 1.5).abs() < 1.0e-12);
        assert!((boxes[1].median - 15.0).abs() < 1.0e-12);
    }

    #[test]
    fn reads_typed_integer_data_groups_and_options_exactly() {
        let options = BoxplotOptions::parse(vec![
            Value::CharArray(CharArray::new_row("Colors")),
            int_tensor(IntegerStorage::U8(vec![0, 1, 0]), vec![1, 3]),
            Value::CharArray(CharArray::new_row("Positions")),
            int_tensor(IntegerStorage::I16(vec![5, 9]), vec![1, 2]),
        ])
        .expect("options");
        let boxes = build_box_stats(
            int_tensor(IntegerStorage::I16(vec![1, 10, 3, 20]), vec![4, 1]),
            Some(int_tensor(
                IntegerStorage::U16(vec![2, 1, 2, 1]),
                vec![4, 1],
            )),
            &options,
        )
        .expect("stats");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].label, "1");
        assert_eq!(boxes[0].position, 5.0);
        assert!((boxes[0].median - 15.0).abs() < 1.0e-12);
        assert_eq!(boxes[1].label, "2");
        assert_eq!(boxes[1].position, 9.0);
        assert!((boxes[1].median - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn scalar_group_label_reads_typed_integer_storage_without_double_mirror() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![42]), vec![1, 1]).unwrap();

        let Some(GroupLabel::Integer(value)) = scalar_group_label(&Value::Tensor(tensor)).unwrap()
        else {
            panic!("expected exact integer label");
        };
        assert_eq!(value, 42);
    }

    #[test]
    fn option_parser_accepts_documented_extreme_and_label_values() {
        let options = BoxplotOptions::parse(vec![
            Value::CharArray(CharArray::new_row("DataLim")),
            Value::Tensor(
                Tensor::new(vec![f64::NEG_INFINITY, f64::INFINITY], vec![1, 2]).expect("tensor"),
            ),
            Value::CharArray(CharArray::new_row("ExtremeMode")),
            Value::CharArray(CharArray::new_row("compress")),
            Value::CharArray(CharArray::new_row("Jitter")),
            Value::Num(0.0),
            Value::CharArray(CharArray::new_row("Symbol")),
            Value::CharArray(CharArray::new_row("square")),
            Value::CharArray(CharArray::new_row("LabelVerbosity")),
            Value::CharArray(CharArray::new_row("majorminor")),
        ])
        .expect("options");
        assert_eq!(options.data_lim, None);
        assert_eq!(options.extreme_mode, ExtremeMode::Compress);
        assert_eq!(options.jitter, 0.0);
        assert_eq!(options.outlier_marker, Some(MarkerStyle::Square));
    }

    #[test]
    fn jitter_zero_keeps_outliers_centered_and_datalim_clips() {
        let options = BoxplotOptions {
            data_lim: Some((0.0, 10.0)),
            jitter: 0.0,
            ..Default::default()
        };
        let stats = BoxStats {
            label: "A".to_string(),
            position: 3.0,
            width: 0.5,
            color: Vec4::ONE,
            q1: 1.0,
            median: 2.0,
            q3: 3.0,
            lower_whisker: 1.0,
            upper_whisker: 3.0,
            outliers: vec![-5.0, 12.0],
        };
        let (x, y) = outlier_points(&stats, &options);
        assert_eq!(x, vec![3.0, 3.0]);
        assert_eq!(y, vec![0.0, 10.0]);
    }

    #[test]
    fn factor_options_validate_known_values() {
        assert!(BoxplotOptions::parse(vec![
            Value::CharArray(CharArray::new_row("FactorDirection")),
            Value::CharArray(CharArray::new_row("data")),
            Value::CharArray(CharArray::new_row("FullFactors")),
            Value::CharArray(CharArray::new_row("off")),
            Value::CharArray(CharArray::new_row("FactorGap")),
            Value::Num(2.0),
            Value::CharArray(CharArray::new_row("FactorSeparator")),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor")),
        ])
        .is_ok());
        assert!(BoxplotOptions::parse(vec![
            Value::CharArray(CharArray::new_row("FactorDirection")),
            Value::CharArray(CharArray::new_row("sideways")),
        ])
        .is_err());
    }

    #[test]
    fn renders_handle_vector_and_tick_labels() {
        let _guard = crate::builtins::plotting::lock_plot_test_context();
        crate::builtins::plotting::reset_plot_state();
        let options = BoxplotOptions::default();
        let boxes = build_box_stats(tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]), None, &options)
            .expect("stats");
        let handles = render_boxplot(&boxes, &options).expect("render");
        assert_eq!(handles.len(), 6);
        let figure = clone_figure(current_figure_handle()).expect("figure");
        let meta = figure.axes_metadata(0).expect("axes metadata");
        assert_eq!(meta.x_tick_labels.as_deref(), Some(&["1".to_string()][..]));
    }
}
