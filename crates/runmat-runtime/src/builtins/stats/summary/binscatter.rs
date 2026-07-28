//! Binned scatter plot compatibility helper.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, ObjectInstance, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::tensor;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_handle};
use crate::builtins::plotting::state::{
    color_limits_snapshot, register_binscatter_handle, render_active_plot, PlotRenderOptions,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "binscatter";
const DEFAULT_FACE_ALPHA: f64 = 1.0;
pub(crate) const MAX_NUM_BINS: usize = 250;
const AUTO_NUM_BINS_CAP: usize = 100;

const OUTPUT_HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the Binscatter chart.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X coordinates or table variable selector.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y coordinates or table variable selector.",
};

const PARAM_TABLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tbl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or timetable.",
};

const PARAM_NBINS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nbins",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Scalar or two-element bin count.",
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
    description: "Binscatter name-value options.",
};

const INPUT_XY: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUT_XY_N: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_NBINS];
const INPUT_XY_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];
const INPUT_TBL_XY: [BuiltinParamDescriptor; 3] = [PARAM_TABLE, PARAM_X, PARAM_Y];
const INPUT_TBL_XY_N: [BuiltinParamDescriptor; 4] = [PARAM_TABLE, PARAM_X, PARAM_Y, PARAM_NBINS];
const INPUT_AX_XY: [BuiltinParamDescriptor; 3] = [PARAM_AX, PARAM_X, PARAM_Y];
const INPUT_AX_XY_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_AX, PARAM_X, PARAM_Y, PARAM_OPTIONS];
const OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLE];

const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "binscatter(x, y)",
        inputs: &INPUT_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(x, y, nbins)",
        inputs: &INPUT_XY_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(___, Name, Value)",
        inputs: &INPUT_XY_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(tbl, xvar, yvar)",
        inputs: &INPUT_TBL_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(tbl, xvar, yvar, nbins)",
        inputs: &INPUT_TBL_XY_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(ax, ___)",
        inputs: &INPUT_AX_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(ax, x, y, Name, Value)",
        inputs: &INPUT_AX_XY_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = binscatter(___)",
        inputs: &INPUT_XY,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = binscatter(x, y, nbins)",
        inputs: &INPUT_XY_N,
        outputs: &OUTPUT_H,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BINSCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:binscatter:InvalidArgument"),
    when: "Inputs, bin controls, or chart options are malformed.",
    message: "binscatter: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BINSCATTER.INTERNAL",
    identifier: Some("RunMat:binscatter:Internal"),
    when: "RunMat cannot compute bins or render the chart.",
    message: "binscatter: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const BINSCATTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Debug)]
struct ParsedInputs {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    rest: Vec<Value>,
}

#[derive(Clone, Debug)]
struct BinscatterOptions {
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<String>,
    num_bins: Option<[usize; 2]>,
    x_limits: Option<(f64, f64)>,
    y_limits: Option<(f64, f64)>,
}

pub(crate) struct BinscatterChart {
    pub(crate) values: Tensor,
    pub(crate) x_bin_edges: Vec<f64>,
    pub(crate) y_bin_edges: Vec<f64>,
    pub(crate) surface: SurfacePlot,
}

fn binscatter_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Num
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

#[runtime_builtin(
    name = "binscatter",
    category = "stats/summary",
    summary = "Visualize point density with a two-dimensional binned scatter plot.",
    keywords = "binscatter,binned scatter,2d histogram,statistics,plotting,density",
    sink = true,
    suppress_auto_output = true,
    type_resolver(binscatter_type),
    descriptor(crate::builtins::stats::summary::binscatter::BINSCATTER_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::binscatter"
)]
pub(crate) async fn binscatter_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let (axes_target, args) = split_leading_axes_handle(args, NAME).map_err(map_plot_error)?;
    apply_axes_target(axes_target, NAME).map_err(map_plot_error)?;

    let args = gather_values(args).await?;
    let mut inputs = parse_inputs(args)?;
    let options = parse_options(std::mem::take(&mut inputs.rest))?;
    let [x_bins, y_bins] = if let Some(num_bins) = options.num_bins {
        num_bins
    } else {
        [
            auto_bin_count(&inputs.x_data, options.x_limits, AUTO_NUM_BINS_CAP)?,
            auto_bin_count(&inputs.y_data, options.y_limits, AUTO_NUM_BINS_CAP)?,
        ]
    };
    let chart = build_binscatter_chart(
        &inputs.x_data,
        &inputs.y_data,
        [x_bins, y_bins],
        options.x_limits,
        options.y_limits,
        options.show_empty_bins,
        options.face_alpha,
        options.display_name.as_deref(),
        color_limits_snapshot(),
    )?;
    render_binscatter(inputs, options, [x_bins, y_bins], chart)
}

async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| invalid(format!("binscatter: {err}")))?,
        );
    }
    Ok(out)
}

fn parse_inputs(mut args: Vec<Value>) -> BuiltinResult<ParsedInputs> {
    if args.len() < 2 {
        return Err(invalid("binscatter: expected x and y inputs"));
    }
    if let Value::Object(object) = &args[0] {
        if crate::builtins::table::is_tabular_object(object) {
            if args.len() < 3 {
                return Err(invalid("binscatter: expected table, xvar, and yvar"));
            }
            let table = match args.remove(0) {
                Value::Object(object) => object,
                _ => unreachable!(),
            };
            let x_selector = args.remove(0);
            let y_selector = args.remove(0);
            let x_data = table_variable_numeric_data(&table, &x_selector, "xvar")?;
            let y_data = table_variable_numeric_data(&table, &y_selector, "yvar")?;
            ensure_compatible_lengths(&x_data, &y_data)?;
            return Ok(ParsedInputs {
                x_data,
                y_data,
                rest: args,
            });
        }
    }

    let x_value = args.remove(0);
    let y_value = args.remove(0);
    let x_data = numeric_data_from_value(x_value, "x")?;
    let y_data = numeric_data_from_value(y_value, "y")?;
    ensure_compatible_lengths(&x_data, &y_data)?;
    Ok(ParsedInputs {
        x_data,
        y_data,
        rest: args,
    })
}

fn table_variable_numeric_data(
    object: &ObjectInstance,
    selector: &Value,
    context: &str,
) -> BuiltinResult<Vec<f64>> {
    let names = crate::builtins::table::table_variable_names_from_object(object)
        .map_err(|err| invalid(err.message))?;
    let name = table_variable_name(selector, &names, context)?;
    let variables =
        crate::builtins::table::table_variables(object).map_err(|err| invalid(err.message))?;
    let value = variables
        .fields
        .get(&name)
        .cloned()
        .ok_or_else(|| invalid(format!("binscatter: table variable '{name}' not found")))?;
    numeric_data_from_value(value, &name)
}

fn table_variable_name(selector: &Value, names: &[String], context: &str) -> BuiltinResult<String> {
    if let Some(name) = tensor::value_to_string(selector) {
        if names.iter().any(|candidate| candidate == &name) {
            return Ok(name);
        }
        return Err(invalid(format!(
            "binscatter: {context} table variable '{name}' not found"
        )));
    }
    if let Some(index) = numeric_selector_index(selector)? {
        if let Some(name) = names.get(index) {
            return Ok(name.clone());
        }
        return Err(invalid(format!(
            "binscatter: {context} table variable index exceeds table width"
        )));
    }
    Err(invalid(format!(
        "binscatter: {context} must be a table variable name or index"
    )))
}

fn numeric_selector_index(selector: &Value) -> BuiltinResult<Option<usize>> {
    if let Some(value) = integer_scalar(selector) {
        let Some(index) = value.try_to_usize().and_then(|value| value.checked_sub(1)) else {
            return Err(invalid(
                "binscatter: table variable index must be a positive integer",
            ));
        };
        return Ok(Some(index));
    }
    let raw = match selector {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    };
    let Some(value) = raw else {
        return Ok(None);
    };
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(invalid(
            "binscatter: table variable index must be a positive integer",
        ));
    }
    Ok(Some(value as usize - 1))
}

fn numeric_data_from_value(value: Value, context: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|_| invalid(format!("binscatter: {context} must be numeric")))?;
    Ok(tensor_values_f64(&tensor))
}

fn ensure_compatible_lengths(x_data: &[f64], y_data: &[f64]) -> BuiltinResult<()> {
    if x_data.len() != y_data.len() {
        return Err(invalid(
            "binscatter: x and y must contain the same number of elements",
        ));
    }
    Ok(())
}

fn parse_options(args: Vec<Value>) -> BuiltinResult<BinscatterOptions> {
    let mut options = BinscatterOptions {
        show_empty_bins: false,
        face_alpha: DEFAULT_FACE_ALPHA,
        display_name: None,
        num_bins: None,
        x_limits: None,
        y_limits: None,
    };
    let mut index = 0;

    if index < args.len() && !is_option_name(&args[index]) {
        options.num_bins = Some(parse_num_bins(&args[index])?);
        index += 1;
    }

    while index < args.len() {
        let key = tensor::value_to_string(&args[index])
            .ok_or_else(|| invalid("binscatter: expected name-value option"))?;
        index += 1;
        if index >= args.len() {
            return Err(invalid(format!(
                "binscatter: missing value for option '{key}'"
            )));
        }
        let value = &args[index];
        index += 1;
        match key.trim().to_ascii_lowercase().as_str() {
            "numbins" => options.num_bins = Some(parse_num_bins(value)?),
            "xlimits" => options.x_limits = Some(parse_limits(value, "XLimits")?),
            "ylimits" => options.y_limits = Some(parse_limits(value, "YLimits")?),
            "showemptybins" => {
                options.show_empty_bins = option_bool(value, "ShowEmptyBins")?;
            }
            "facealpha" => {
                options.face_alpha = option_scalar(value, "FaceAlpha")?;
                validate_face_alpha(options.face_alpha)?;
            }
            "displayname" => {
                options.display_name = Some(
                    tensor::value_to_string(value)
                        .ok_or_else(|| invalid("binscatter: DisplayName must be text"))?,
                );
            }
            "displaystyle" => {
                let display_style = tensor::value_to_string(value)
                    .ok_or_else(|| invalid("binscatter: DisplayStyle must be a string"))?;
                if !display_style.trim().eq_ignore_ascii_case("tile") {
                    return Err(invalid("binscatter: only DisplayStyle 'tile' is supported"));
                }
            }
            "normalization" | "binwidth" | "xbinwidth" | "ybinwidth" | "binmethod"
            | "xbinmethod" | "ybinmethod" | "binlimits" | "xbinlimits" | "ybinlimits"
            | "xbinedges" | "ybinedges" => {
                return Err(invalid(format!(
                    "binscatter: option '{key}' is not supported by MATLAB binscatter"
                )));
            }
            other => {
                return Err(invalid(format!(
                    "binscatter: unrecognised option '{other}'"
                )));
            }
        }
    }

    Ok(options)
}

fn is_option_name(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

pub(crate) fn parse_num_bins(value: &Value) -> BuiltinResult<[usize; 2]> {
    if let Some(values) = integer_values(value) {
        let bins = match values.as_slice() {
            [n] => [positive_bin_count_integer(n, "NumBins")?; 2],
            [nx, ny] => [
                positive_bin_count_integer(nx, "NumBins")?,
                positive_bin_count_integer(ny, "NumBins")?,
            ],
            _ => {
                return Err(invalid(
                    "binscatter: NumBins must be a scalar or two-element vector",
                ))
            }
        };
        return Ok(bins);
    }
    let values = numeric_vector(value, "NumBins")?;
    let bins = match values.as_slice() {
        [n] => [positive_bin_count(*n, "NumBins")?; 2],
        [nx, ny] => [
            positive_bin_count(*nx, "NumBins")?,
            positive_bin_count(*ny, "NumBins")?,
        ],
        _ => {
            return Err(invalid(
                "binscatter: NumBins must be a scalar or two-element vector",
            ))
        }
    };
    Ok(bins)
}

pub(crate) fn parse_limits(value: &Value, name: &str) -> BuiltinResult<(f64, f64)> {
    let values = numeric_vector(value, name)?;
    if values.len() != 2 {
        return Err(invalid(format!(
            "binscatter: {name} must be a two-element vector"
        )));
    }
    validate_limits(values[0], values[1], name)
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid(format!("binscatter: {name} must be numeric")))?;
    Ok(tensor_values_f64(&tensor))
}

fn positive_bin_count(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(invalid(format!(
            "binscatter: {name} entries must be positive integers"
        )));
    }
    let count = value as usize;
    if count > MAX_NUM_BINS {
        return Err(invalid(format!(
            "binscatter: {name} entries must be no greater than {MAX_NUM_BINS}"
        )));
    }
    Ok(count)
}

fn positive_bin_count_integer(value: &IntValue, name: &str) -> BuiltinResult<usize> {
    let Some(count) = value.try_to_usize() else {
        return Err(invalid(format!(
            "binscatter: {name} entries must be positive integers"
        )));
    };
    if count == 0 {
        return Err(invalid(format!(
            "binscatter: {name} entries must be positive integers"
        )));
    }
    if count > MAX_NUM_BINS {
        return Err(invalid(format!(
            "binscatter: {name} entries must be no greater than {MAX_NUM_BINS}"
        )));
    }
    Ok(count)
}

fn validate_limits(lo: f64, hi: f64, name: &str) -> BuiltinResult<(f64, f64)> {
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return Err(invalid(format!(
            "binscatter: {name} must contain finite increasing limits"
        )));
    }
    Ok((lo, hi))
}

pub(crate) fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(value) = integer_scalar(value) {
        return integer_bool(&value, name);
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => numeric_bool(*value, name),
        _ => {
            if let Some(text) = tensor::value_to_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" => Ok(true),
                    "off" | "false" | "0" => Ok(false),
                    _ => Err(invalid(format!("binscatter: {name} must be logical"))),
                }
            } else {
                Err(invalid(format!("binscatter: {name} must be logical")))
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
        Err(invalid(format!("binscatter: {name} must be 0 or 1")))
    }
}

fn integer_bool(value: &IntValue, name: &str) -> BuiltinResult<bool> {
    match value.try_to_usize() {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => Err(invalid(format!("binscatter: {name} must be 0 or 1"))),
    }
}

fn option_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(value) = integer_scalar(value) {
        return Ok(value.to_f64());
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        _ => Err(invalid(format!("binscatter: {name} must be a scalar"))),
    }
}

fn integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn integer_values(value: &Value) -> Option<Vec<IntValue>> {
    match value {
        Value::Int(value) => Some(vec![value.clone()]),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| storage.exact_values()),
        _ => None,
    }
}

fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

pub(crate) fn validate_face_alpha(value: f64) -> BuiltinResult<()> {
    if !(0.0..=1.0).contains(&value) || !value.is_finite() {
        return Err(invalid(
            "binscatter: FaceAlpha must be in the interval [0,1]",
        ));
    }
    Ok(())
}

pub(crate) fn build_binscatter_chart(
    x_data: &[f64],
    y_data: &[f64],
    num_bins: [usize; 2],
    x_limits: Option<(f64, f64)>,
    y_limits: Option<(f64, f64)>,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<&str>,
    color_limits: Option<(f64, f64)>,
) -> BuiltinResult<BinscatterChart> {
    ensure_compatible_lengths(x_data, y_data)?;
    validate_bin_counts(num_bins)?;
    validate_face_alpha(face_alpha)?;

    let x_limits = derive_axis_limits(x_data, x_limits, "XLimits")?;
    let y_limits = derive_axis_limits(y_data, y_limits, "YLimits")?;
    let x_edges = linspace(x_limits.0, x_limits.1, num_bins[0] + 1);
    let y_edges = linspace(y_limits.0, y_limits.1, num_bins[1] + 1);
    let cell_count = num_bins[0]
        .checked_mul(num_bins[1])
        .ok_or_else(|| invalid("binscatter: NumBins product is too large"))?;
    let mut counts = vec![0.0; cell_count];
    for (&x, &y) in x_data.iter().zip(y_data.iter()) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        if let (Some(ix), Some(iy)) = (find_bin_index(x, &x_edges), find_bin_index(y, &y_edges)) {
            counts[ix + iy * num_bins[0]] += 1.0;
        }
    }
    let values = Tensor::new(counts.clone(), vec![num_bins[0], num_bins[1]])
        .map_err(|err| internal(format!("binscatter: {err}")))?;
    let surface = build_surface_from_counts(
        &counts,
        &x_edges,
        &y_edges,
        num_bins,
        show_empty_bins,
        face_alpha,
        display_name,
        color_limits,
    )?;
    Ok(BinscatterChart {
        values,
        x_bin_edges: x_edges,
        y_bin_edges: y_edges,
        surface,
    })
}

pub(crate) fn auto_bin_count(
    data: &[f64],
    limits: Option<(f64, f64)>,
    cap: usize,
) -> BuiltinResult<usize> {
    let limits = derive_axis_limits(data, limits, "Limits")?;
    let filtered = data
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value >= limits.0 && *value <= limits.1)
        .collect::<Vec<_>>();
    if filtered.len() < 2 {
        return Ok(1);
    }
    let width = scott_width(&filtered);
    if !width.is_finite() || width <= 0.0 {
        return Ok(1);
    }
    let bins = ((limits.1 - limits.0) / width).ceil().max(1.0) as usize;
    Ok(bins.clamp(1, cap.min(MAX_NUM_BINS)))
}

fn validate_bin_counts(num_bins: [usize; 2]) -> BuiltinResult<()> {
    for count in num_bins {
        if count == 0 || count > MAX_NUM_BINS {
            return Err(invalid(format!(
                "binscatter: NumBins entries must be in the range 1..={MAX_NUM_BINS}"
            )));
        }
    }
    Ok(())
}

fn derive_axis_limits(
    data: &[f64],
    explicit: Option<(f64, f64)>,
    name: &str,
) -> BuiltinResult<(f64, f64)> {
    if let Some((lo, hi)) = explicit {
        return validate_limits(lo, hi, name);
    }
    let mut min_value: Option<f64> = None;
    let mut max_value: Option<f64> = None;
    for value in data.iter().copied().filter(|value| value.is_finite()) {
        min_value = Some(min_value.map_or(value, |current| current.min(value)));
        max_value = Some(max_value.map_or(value, |current| current.max(value)));
    }
    let (mut lo, mut hi) = match (min_value, max_value) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (0.0, 1.0),
    };
    if lo == hi {
        let pad = lo.abs().max(1.0) * 0.5;
        lo -= pad;
        hi += pad;
    }
    Ok((lo, hi))
}

fn scott_width(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    let sigma = variance.sqrt();
    if sigma <= 0.0 {
        0.0
    } else {
        3.5 * sigma / (values.len() as f64).powf(1.0 / 3.0)
    }
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        out.push(start + step * idx as f64);
    }
    if let Some(last) = out.last_mut() {
        *last = stop;
    }
    out
}

fn find_bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    match edges.binary_search_by(|edge| edge.partial_cmp(&value).unwrap_or(Ordering::Less)) {
        Ok(index) => {
            if index == 0 {
                Some(0)
            } else if index < edges.len() - 1 {
                Some(index)
            } else {
                Some(edges.len() - 2)
            }
        }
        Err(index) => {
            if index == 0 || index > edges.len() - 1 {
                None
            } else {
                Some(index - 1)
            }
        }
    }
}

fn build_surface_from_counts(
    counts: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
    num_bins: [usize; 2],
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<&str>,
    color_limits: Option<(f64, f64)>,
) -> BuiltinResult<SurfacePlot> {
    let mut z_grid = vec![vec![0.0; num_bins[1]]; num_bins[0]];
    for (ix, row) in z_grid.iter_mut().enumerate().take(num_bins[0]) {
        for (iy, cell) in row.iter_mut().enumerate().take(num_bins[1]) {
            let mut value = counts[ix + iy * num_bins[0]];
            if !show_empty_bins && value == 0.0 {
                value = f64::NAN;
            }
            *cell = value;
        }
    }
    let mut surface = SurfacePlot::new(bin_centers(x_edges), bin_centers(y_edges), z_grid)
        .map_err(|err| internal(format!("binscatter: {err}")))?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::None)
        .with_alpha(face_alpha as f32);
    if let Some(display_name) = display_name {
        surface = surface.with_label(display_name.to_string());
    }
    if color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }
    Ok(surface)
}

fn bin_centers(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

fn render_binscatter(
    inputs: ParsedInputs,
    options: BinscatterOptions,
    num_bins: [usize; 2],
    chart: BinscatterChart,
) -> BuiltinResult<f64> {
    let mut surface = Some(chart.surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "Binscatter Plot",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        move |figure, axes| {
            let plot_index = figure.add_surface_plot_on_axes(
                surface.take().expect("binscatter plot consumed once"),
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
    let handle = register_binscatter_handle(
        figure_handle,
        axes,
        plot_index,
        chart.values,
        chart.x_bin_edges,
        chart.y_bin_edges,
        inputs.x_data,
        inputs.y_data,
        num_bins,
        options.num_bins.is_none(),
        options.x_limits,
        options.y_limits,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use crate::builtins::table::table_from_columns;
    use runmat_builtins::IntegerStorage;
    use runmat_plot::plots::figure::PlotElement;

    fn vec_tensor(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).unwrap();
        tensor.data.clear();
        Value::Tensor(tensor)
    }

    #[test]
    fn binscatter_parses_typed_integer_options_exactly() {
        assert_eq!(
            parse_num_bins(&int_tensor(IntegerStorage::U16(vec![12, 8]), vec![1, 2])).unwrap(),
            [12, 8]
        );
        assert_eq!(
            parse_num_bins(&Value::Int(IntValue::U8(9))).unwrap(),
            [9, 9]
        );
        assert_eq!(
            numeric_selector_index(&int_tensor(IntegerStorage::I32(vec![2]), vec![1, 1])).unwrap(),
            Some(1)
        );
        assert!(option_bool(
            &int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            "ShowEmptyBins"
        )
        .unwrap());
        assert_eq!(
            option_scalar(
                &int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]),
                "FaceAlpha",
            )
            .unwrap(),
            IntValue::U64(9_007_199_254_740_993).to_f64()
        );
    }

    #[test]
    fn binscatter_rejects_invalid_typed_integer_options() {
        let err =
            parse_num_bins(&int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1])).unwrap_err();
        assert!(
            err.message().contains("positive integers"),
            "{}",
            err.message()
        );

        let err =
            parse_num_bins(&int_tensor(IntegerStorage::U16(vec![251]), vec![1, 1])).unwrap_err();
        assert!(
            err.message().contains("no greater than 250"),
            "{}",
            err.message()
        );

        let err = option_bool(
            &int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            "ShowEmptyBins",
        )
        .unwrap_err();
        assert!(err.message().contains("0 or 1"), "{}", err.message());
    }

    #[test]
    fn binscatter_counts_and_registers_chart_state() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(binscatter_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let props = crate::builtins::plotting::properties::get_properties(
            crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
                .unwrap(),
            Some("Values"),
            NAME,
        )
        .unwrap();
        let Value::Tensor(values) = props else {
            panic!("expected Values tensor");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(values.data, vec![2.0, 0.0, 0.0, 2.0]);

        let figure = clone_figure(current_figure_handle()).unwrap();
        let plot = figure.plots().next().unwrap();
        let PlotElement::Surface(surface) = plot else {
            panic!("expected surface plot");
        };
        assert!(surface.image_mode);
        assert!(surface.flatten_z);
        assert_eq!(surface.x_data, vec![0.25, 0.75]);
        assert_eq!(surface.y_data, vec![0.25, 0.75]);
    }

    #[test]
    fn binscatter_accepts_limits_show_empty_bins_and_face_alpha() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(binscatter_builtin(vec![
            vec_tensor(&[0.0, 1.0, 2.0]),
            vec_tensor(&[0.0, 1.0, 2.0]),
            Value::String("NumBins".into()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
            Value::String("XLimits".into()),
            Value::Tensor(Tensor::new(vec![0.0, 2.0], vec![1, 2]).unwrap()),
            Value::String("YLimits".into()),
            Value::Tensor(Tensor::new(vec![0.0, 2.0], vec![1, 2]).unwrap()),
            Value::String("ShowEmptyBins".into()),
            Value::String("on".into()),
            Value::String("FaceAlpha".into()),
            Value::Num(0.5),
            Value::String("DisplayName".into()),
            Value::String("density".into()),
        ]))
        .unwrap();
        let face_alpha = crate::builtins::plotting::properties::get_properties(
            crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
                .unwrap(),
            Some("FaceAlpha"),
            NAME,
        )
        .unwrap();
        assert!(matches!(face_alpha, Value::Num(value) if (value - 0.5).abs() < 1.0e-12));
    }

    #[test]
    fn binscatter_supports_table_variable_inputs() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let table = table_from_columns(
            vec!["x".to_string(), "y".to_string()],
            vec![
                Tensor::new(vec![0.0, 0.2, 0.8, 1.0], vec![4, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![0.0, 0.1, 0.9, 1.0], vec![4, 1])
                    .map(Value::Tensor)
                    .unwrap(),
            ],
        )
        .unwrap();
        let handle = futures::executor::block_on(binscatter_builtin(vec![
            table,
            Value::String("x".into()),
            Value::String("y".into()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let props = crate::builtins::plotting::properties::get_properties(
            crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
                .unwrap(),
            Some("Values"),
            NAME,
        )
        .unwrap();
        let Value::Tensor(values) = props else {
            panic!("expected Values tensor");
        };
        assert_eq!(values.data, vec![2.0, 0.0, 0.0, 2.0]);
    }
}
