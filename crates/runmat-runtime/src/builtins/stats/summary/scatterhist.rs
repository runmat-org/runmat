//! Scatter plot with marginal histograms.

use glam::Vec4;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, IntValue, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{
    bar::Orientation, BarChart, LinePlot, LineStyle, MarkerStyle, ScatterPlot,
};

use crate::builtins::common::tensor;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    configure_subplot, current_axes_state, encode_axes_handle, render_active_plot,
    select_axes_for_figure, set_legend_for_axes, FigureHandle, PlotRenderOptions,
};
use crate::builtins::plotting::style::{parse_color_value, value_as_string, LineStyleParseOptions};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "scatterhist";
const MAX_BINS: usize = 250;
const DEFAULT_MARKER_SIZE: f32 = 6.0;
const DEFAULT_LINE_WIDTH: f32 = 0.5;
const DEFAULT_AUTO_BIN_CAP: usize = 100;
const MAX_GROUPS: usize = 256;
const GROUP_COLORS: [Vec4; 7] = [
    Vec4::new(0.0, 0.447, 0.741, 1.0),
    Vec4::new(0.85, 0.325, 0.098, 1.0),
    Vec4::new(0.929, 0.694, 0.125, 1.0),
    Vec4::new(0.494, 0.184, 0.556, 1.0),
    Vec4::new(0.466, 0.674, 0.188, 1.0),
    Vec4::new(0.301, 0.745, 0.933, 1.0),
    Vec4::new(0.635, 0.078, 0.184, 1.0),
];

const OUTPUT_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Three axes handles: scatter, x marginal, and y marginal.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X observations.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y observations.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Scatterhist name-value options.",
};

const INPUT_XY: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUT_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_AX];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "scatterhist(x, y)",
        inputs: &INPUT_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "scatterhist(x, y, Name, Value)",
        inputs: &INPUT_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterhist(___)",
        inputs: &INPUT_OPTIONS,
        outputs: &OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERHIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatterhist:InvalidArgument"),
    when: "Observation vectors or name-value options are malformed.",
    message: "scatterhist: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERHIST.INTERNAL",
    identifier: Some("RunMat:scatterhist:Internal"),
    when: "RunMat cannot allocate or render the scatterhist chart.",
    message: "scatterhist: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const SCATTERHIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Debug)]
struct ScatterhistOptions {
    bins: Option<[usize; 2]>,
    group: Option<Vec<Option<String>>>,
    kernel: KernelMode,
    legend: Option<bool>,
    plot_group: bool,
    hist_style: Option<HistStyle>,
    location: Location,
    direction: Direction,
    colors: Option<Vec<Vec4>>,
    markers: Vec<MarkerStyle>,
    marker_sizes: Vec<f32>,
    line_widths: Vec<f32>,
    line_styles: Vec<LineStyle>,
    bandwidths: Vec<[f64; 2]>,
    parent: Option<FigureHandle>,
}

impl Default for ScatterhistOptions {
    fn default() -> Self {
        Self {
            bins: None,
            group: None,
            kernel: KernelMode::Off,
            legend: None,
            plot_group: true,
            hist_style: None,
            location: Location::SouthWest,
            direction: Direction::In,
            colors: None,
            markers: vec![MarkerStyle::Circle],
            marker_sizes: vec![DEFAULT_MARKER_SIZE],
            line_widths: vec![DEFAULT_LINE_WIDTH],
            line_styles: vec![LineStyle::Solid],
            bandwidths: Vec::new(),
            parent: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelMode {
    Off,
    On,
    Overlay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HistStyle {
    Bar,
    Stairs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Location {
    NorthEast,
    SouthEast,
    SouthWest,
    NorthWest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Out,
    In,
}

#[derive(Clone, Debug)]
struct GroupSeries {
    label: String,
    scatter_x: Vec<f64>,
    scatter_y: Vec<f64>,
    marginal_x: Vec<f64>,
    marginal_y: Vec<f64>,
    style: SeriesStyle,
}

#[derive(Clone, Debug)]
struct SeriesStyle {
    color: Vec4,
    marker: MarkerStyle,
    marker_size: f32,
    line_width: f32,
    line_style: LineStyle,
    bandwidth: Option<[f64; 2]>,
}

#[derive(Clone, Copy, Debug)]
struct LayoutAxes {
    scatter: usize,
    xhist: usize,
    yhist: usize,
    empty: usize,
    x_marginal_scale: f64,
    y_marginal_scale: f64,
}

fn scatterhist_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![Some(3), Some(1)]),
    }
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

#[runtime_builtin(
    name = "scatterhist",
    category = "stats/summary",
    summary = "Create a scatter plot with marginal histograms.",
    keywords = "scatterhist,scatter,histogram,marginal,statistics,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(scatterhist_type),
    descriptor(crate::builtins::stats::summary::scatterhist::SCATTERHIST_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::scatterhist"
)]
pub(crate) async fn scatterhist_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(invalid("scatterhist: expected x and y inputs"));
    }
    let args = gather_values(args).await?;
    let (x, y, rest) = parse_xy_and_rest(args)?;
    let options = parse_options(rest, x.len())?;
    let series = grouped_series(&x, &y, &options)?;
    let axes = render_scatterhist(&series, &options)?;
    Ok(Value::Tensor(
        Tensor::new(axes.to_vec(), vec![3, 1])
            .map_err(|err| internal(format!("scatterhist: {err}")))?,
    ))
}

async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| invalid(format!("scatterhist: {err}")))?,
        );
    }
    Ok(out)
}

fn parse_xy_and_rest(args: Vec<Value>) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<Value>)> {
    let mut iter = args.into_iter();
    let x = numeric_vector(iter.next().unwrap(), "x")?;
    let y = numeric_vector(iter.next().unwrap(), "y")?;
    if x.len() != y.len() {
        return Err(invalid(
            "scatterhist: x and y must contain the same number of elements",
        ));
    }
    if x.is_empty() {
        return Err(invalid("scatterhist: x and y must not be empty"));
    }
    Ok((x, y, iter.collect()))
}

fn numeric_vector(value: Value, name: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|_| invalid(format!("scatterhist: {name} must be numeric")))?;
    if tensor.shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(format!("scatterhist: {name} must be a vector")));
    }
    Ok(tensor_values_f64(&tensor))
}

fn parse_options(args: Vec<Value>, observation_count: usize) -> BuiltinResult<ScatterhistOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid("scatterhist: name-value options must be paired"));
    }
    let mut options = ScatterhistOptions::default();
    for pair in args.chunks_exact(2) {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| invalid("scatterhist: option names must be text"))?;
        match key.trim().to_ascii_lowercase().as_str() {
            "nbins" => options.bins = Some(parse_bins(&pair[1])?),
            "group" => options.group = Some(group_labels(&pair[1], observation_count)?),
            "kernel" => options.kernel = parse_kernel(&pair[1])?,
            "legend" => options.legend = Some(option_bool(&pair[1], "Legend")?),
            "plotgroup" => options.plot_group = option_bool(&pair[1], "PlotGroup")?,
            "location" => options.location = parse_location(&pair[1])?,
            "direction" => options.direction = parse_direction(&pair[1])?,
            "color" => options.colors = Some(parse_color_cycle(&pair[1])?),
            "marker" => options.markers = parse_marker_cycle(&pair[1])?,
            "markersize" => {
                options.marker_sizes = nonnegative_scalar_cycle(&pair[1], "MarkerSize")?
            }
            "linewidth" => options.line_widths = nonnegative_scalar_cycle(&pair[1], "LineWidth")?,
            "linestyle" => options.line_styles = parse_line_style_cycle(&pair[1])?,
            "style" => {
                let (kernel, style) = parse_style(&pair[1])?;
                options.kernel = kernel;
                options.hist_style = Some(style);
            }
            "bandwidth" => options.bandwidths = parse_bandwidth_cycle(&pair[1])?,
            "parent" => options.parent = Some(parse_parent(&pair[1])?),
            other => {
                return Err(invalid(format!(
                    "scatterhist: unsupported option '{other}'"
                )))
            }
        }
    }
    Ok(options)
}

fn parse_bins(value: &Value) -> BuiltinResult<[usize; 2]> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid("scatterhist: NBins must be numeric"))?;
    if let Some(values) = integer_values(&tensor) {
        return match values.as_slice() {
            [n] => Ok([bin_count_integer(n)?, bin_count_integer(n)?]),
            [nx, ny] => Ok([bin_count_integer(nx)?, bin_count_integer(ny)?]),
            _ => Err(invalid(
                "scatterhist: NBins must be a scalar or two-element vector",
            )),
        };
    }
    let values = tensor_values_f64(&tensor);
    match values.as_slice() {
        [n] => Ok([bin_count(*n)?, bin_count(*n)?]),
        [nx, ny] => Ok([bin_count(*nx)?, bin_count(*ny)?]),
        _ => Err(invalid(
            "scatterhist: NBins must be a scalar or two-element vector",
        )),
    }
}

fn bin_count(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 2.0 || value.fract() != 0.0 {
        return Err(invalid(
            "scatterhist: NBins entries must be integers greater than or equal to 2",
        ));
    }
    if value > MAX_BINS as f64 {
        return Err(invalid(format!(
            "scatterhist: NBins entries must be no greater than {MAX_BINS}"
        )));
    }
    let value = value as usize;
    if value > MAX_BINS {
        return Err(invalid(format!(
            "scatterhist: NBins entries must be no greater than {MAX_BINS}"
        )));
    }
    Ok(value)
}

fn bin_count_integer(value: &IntValue) -> BuiltinResult<usize> {
    let Some(value) = value.try_to_usize() else {
        return Err(invalid(
            "scatterhist: NBins entries must be integers greater than or equal to 2",
        ));
    };
    if value < 2 {
        return Err(invalid(
            "scatterhist: NBins entries must be integers greater than or equal to 2",
        ));
    }
    if value > MAX_BINS {
        return Err(invalid(format!(
            "scatterhist: NBins entries must be no greater than {MAX_BINS}"
        )));
    }
    Ok(value)
}

fn parse_bandwidth_cycle(value: &Value) -> BuiltinResult<Vec<[f64; 2]>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid("scatterhist: Bandwidth must be numeric"))?;
    let values = tensor_values_f64(&tensor);
    if tensor.rows > 0 && tensor.cols == 2 {
        let mut bandwidths = Vec::with_capacity(tensor.rows);
        for row in 0..tensor.rows {
            bandwidths.push([
                bandwidth_value(values[row])?,
                bandwidth_value(values[tensor.rows + row])?,
            ]);
        }
        return Ok(bandwidths);
    }
    match values.as_slice() {
        [bw] => {
            let bw = bandwidth_value(*bw)?;
            Ok(vec![[bw, bw]])
        }
        [bx, by] => Ok(vec![[bandwidth_value(*bx)?, bandwidth_value(*by)?]]),
        _ => Err(invalid(
            "scatterhist: Bandwidth must be a scalar, two-element vector, or n-by-2 matrix",
        )),
    }
}

fn bandwidth_value(value: f64) -> BuiltinResult<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid("scatterhist: Bandwidth values must be positive"));
    }
    Ok(value)
}

fn nonnegative_scalar_cycle(value: &Value, name: &str) -> BuiltinResult<Vec<f32>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid(format!("scatterhist: {name} must be numeric")))?;
    let values = tensor_values_f64(&tensor);
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(format!(
            "scatterhist: {name} values must be nonnegative"
        )));
    }
    Ok(values.iter().map(|value| *value as f32).collect())
}

fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(text) = value_as_string(value) {
        return match text.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" => Ok(true),
            "off" | "false" | "no" => Ok(false),
            other => Err(invalid(format!(
                "scatterhist: unsupported {name} value '{other}'"
            ))),
        };
    }
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid(format!("scatterhist: {name} must be logical")))?;
    if tensor::tensor_element_len(&tensor) != 1 {
        return Err(invalid(format!(
            "scatterhist: {name} must be a logical scalar"
        )));
    }
    if let Some(value) = tensor
        .integer_storage()
        .and_then(|storage| storage.value_at(0))
    {
        return match value.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid(format!("scatterhist: {name} must be 0 or 1"))),
        };
    }
    match tensor::tensor_value_f64(&tensor, 0) {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(invalid(format!("scatterhist: {name} must be 0 or 1"))),
    }
}

fn parse_kernel(value: &Value) -> BuiltinResult<KernelMode> {
    if let Some(text) = value_as_string(value) {
        return match text.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" => Ok(KernelMode::On),
            "off" | "false" | "no" => Ok(KernelMode::Off),
            "overlay" => Ok(KernelMode::Overlay),
            other => Err(invalid(format!(
                "scatterhist: unsupported Kernel value '{other}'"
            ))),
        };
    }
    Ok(if option_bool(value, "Kernel")? {
        KernelMode::On
    } else {
        KernelMode::Off
    })
}

fn parse_style(value: &Value) -> BuiltinResult<(KernelMode, HistStyle)> {
    let text = value_as_string(value).ok_or_else(|| invalid("scatterhist: Style must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "bar" | "histogram" => Ok((KernelMode::Off, HistStyle::Bar)),
        "stairs" => Ok((KernelMode::Off, HistStyle::Stairs)),
        other => Err(invalid(format!(
            "scatterhist: unsupported Style value '{other}'"
        ))),
    }
}

fn parse_location(value: &Value) -> BuiltinResult<Location> {
    let text =
        value_as_string(value).ok_or_else(|| invalid("scatterhist: Location must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "northeast" => Ok(Location::NorthEast),
        "southeast" => Ok(Location::SouthEast),
        "southwest" => Ok(Location::SouthWest),
        "northwest" => Ok(Location::NorthWest),
        other => Err(invalid(format!(
            "scatterhist: unsupported Location value '{other}'"
        ))),
    }
}

fn parse_direction(value: &Value) -> BuiltinResult<Direction> {
    let text =
        value_as_string(value).ok_or_else(|| invalid("scatterhist: Direction must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "out" => Ok(Direction::Out),
        "in" => Ok(Direction::In),
        other => Err(invalid(format!(
            "scatterhist: unsupported Direction value '{other}'"
        ))),
    }
}

fn parse_marker_cycle(value: &Value) -> BuiltinResult<Vec<MarkerStyle>> {
    let entries = style_text_entries(value, "Marker")?;
    let mut markers = Vec::new();
    for entry in entries {
        let trimmed = entry.trim();
        let marker_chars = trimmed
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<Vec<_>>();
        if marker_chars.len() > 1
            && marker_chars
                .iter()
                .all(|ch| marker_from_text(&ch.to_string()).is_some())
        {
            markers.extend(
                marker_chars
                    .iter()
                    .filter_map(|ch| marker_from_text(&ch.to_string())),
            );
        } else if let Some(marker) = marker_from_text(trimmed) {
            markers.push(marker);
        } else {
            return Err(invalid(format!(
                "scatterhist: unsupported Marker value '{trimmed}'"
            )));
        }
    }
    if markers.is_empty() {
        return Err(invalid("scatterhist: Marker must not be empty"));
    }
    Ok(markers)
}

fn marker_from_text(text: &str) -> Option<MarkerStyle> {
    match text {
        "o" | "circle" => Some(MarkerStyle::Circle),
        "+" | "plus" => Some(MarkerStyle::Plus),
        "*" | "star" => Some(MarkerStyle::Star),
        "x" | "cross" => Some(MarkerStyle::Cross),
        "s" | "square" => Some(MarkerStyle::Square),
        "d" | "diamond" => Some(MarkerStyle::Diamond),
        "^" | "triangle" => Some(MarkerStyle::Triangle),
        _ => None,
    }
}

fn parse_line_style_cycle(value: &Value) -> BuiltinResult<Vec<LineStyle>> {
    let entries = style_text_entries(value, "LineStyle")?;
    let mut styles = Vec::with_capacity(entries.len());
    for entry in entries {
        styles.push(parse_line_style_text(&entry)?);
    }
    if styles.is_empty() {
        return Err(invalid("scatterhist: LineStyle must not be empty"));
    }
    Ok(styles)
}

fn parse_line_style_text(text: &str) -> BuiltinResult<LineStyle> {
    match text.trim().to_ascii_lowercase().as_str() {
        "-" | "solid" => Ok(LineStyle::Solid),
        "--" | "dashed" => Ok(LineStyle::Dashed),
        ":" | "dotted" => Ok(LineStyle::Dotted),
        "-." | "dashdot" => Ok(LineStyle::DashDot),
        "none" => Ok(LineStyle::None),
        other => Err(invalid(format!(
            "scatterhist: unsupported LineStyle value '{other}'"
        ))),
    }
}

fn style_text_entries(value: &Value, option_name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(chars) if chars.rows > 1 => Ok((0..chars.rows)
            .map(|row| {
                let start = row * chars.cols;
                let end = start + chars.cols;
                chars.data[start..end]
                    .iter()
                    .collect::<String>()
                    .trim()
                    .to_string()
            })
            .collect()),
        Value::CharArray(chars) => Ok(vec![chars.data.iter().collect()]),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|entry| {
                value_as_string(entry).ok_or_else(|| {
                    invalid(format!(
                        "scatterhist: {option_name} cell entries must be text"
                    ))
                })
            })
            .collect(),
        _ => Err(invalid(format!(
            "scatterhist: {option_name} must be text, string array, or cell text"
        ))),
    }
}

fn parse_color_cycle(value: &Value) -> BuiltinResult<Vec<Vec4>> {
    if let Some(text) = scalar_text(value) {
        let tokens = text
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<Vec<_>>();
        if tokens.len() > 1 && tokens.iter().all(|ch| color_char(*ch).is_some()) {
            return tokens
                .into_iter()
                .map(|ch| {
                    color_char(ch).ok_or_else(|| {
                        invalid(format!("scatterhist: unsupported Color character '{ch}'"))
                    })
                })
                .collect();
        }
        return Ok(vec![parse_color_text_or_value(value)?]);
    }
    match value {
        Value::StringArray(array) => array
            .data
            .iter()
            .map(|entry| parse_color_text(entry))
            .collect(),
        Value::Cell(cell) => cell.data.iter().map(parse_color_text_or_value).collect(),
        _ => {
            let tensor = tensor::value_to_tensor(value)
                .map_err(|_| invalid("scatterhist: Color must be text or an RGB matrix"))?;
            let values = tensor_values_f64(&tensor);
            if tensor.rows > 0 && tensor.cols == 3 {
                let mut colors = Vec::with_capacity(tensor.rows);
                for row in 0..tensor.rows {
                    colors.push(rgb_color(
                        values[row],
                        values[tensor.rows + row],
                        values[2 * tensor.rows + row],
                    )?);
                }
                return Ok(colors);
            }
            if values.len() == 3 {
                return Ok(vec![rgb_color(values[0], values[1], values[2])?]);
            }
            Err(invalid(
                "scatterhist: Color must be a color token, RGB triplet, or n-by-3 RGB matrix",
            ))
        }
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn parse_color_text_or_value(value: &Value) -> BuiltinResult<Vec4> {
    if let Some(text) = scalar_text(value) {
        return parse_color_text(&text);
    }
    parse_color_value(&LineStyleParseOptions::generic(NAME), value)
        .map_err(|err| invalid(format!("scatterhist: {}", err.message)))
}

fn parse_color_text(text: &str) -> BuiltinResult<Vec4> {
    let trimmed = text.trim();
    if let Some(color) =
        color_char(trimmed.chars().next().unwrap_or('\0')).filter(|_| trimmed.chars().count() == 1)
    {
        return Ok(color);
    }
    parse_color_value(
        &LineStyleParseOptions::generic(NAME),
        &Value::String(trimmed.to_string()),
    )
    .map_err(|err| invalid(format!("scatterhist: {}", err.message)))
}

fn color_char(ch: char) -> Option<Vec4> {
    match ch.to_ascii_lowercase() {
        'y' => Some(Vec4::new(1.0, 1.0, 0.0, 1.0)),
        'm' => Some(Vec4::new(1.0, 0.0, 1.0, 1.0)),
        'c' => Some(Vec4::new(0.0, 1.0, 1.0, 1.0)),
        'r' => Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        'g' => Some(Vec4::new(0.0, 1.0, 0.0, 1.0)),
        'b' => Some(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        'w' => Some(Vec4::new(1.0, 1.0, 1.0, 1.0)),
        'k' => Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        _ => None,
    }
}

fn rgb_color(r: f64, g: f64, b: f64) -> BuiltinResult<Vec4> {
    if !(0.0..=1.0).contains(&r) || !(0.0..=1.0).contains(&g) || !(0.0..=1.0).contains(&b) {
        return Err(invalid(
            "scatterhist: RGB color values must be in the range [0,1]",
        ));
    }
    Ok(Vec4::new(r as f32, g as f32, b as f32, 1.0))
}

fn parse_parent(value: &Value) -> BuiltinResult<FigureHandle> {
    match resolve_plot_handle(value, NAME) {
        Ok(PlotHandle::Figure(handle)) => Ok(handle),
        Ok(PlotHandle::Axes(handle, _)) => Ok(handle),
        Ok(_) => Err(invalid(
            "scatterhist: Parent must be a figure or axes handle in this build",
        )),
        Err(err) => Err(invalid(format!("scatterhist: {}", err.message))),
    }
}

fn group_labels(value: &Value, expected_len: usize) -> BuiltinResult<Vec<Option<String>>> {
    match value {
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == expected_len => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok(storage
                    .exact_values()
                    .into_iter()
                    .map(|value| Some(integer_label(&value)))
                    .collect());
            }
            Ok(tensor
                .materialize_f64()
                .into_iter()
                .map(|value| value.is_finite().then(|| number_label(value)))
                .collect())
        }
        Value::StringArray(array) if array.data.len() == expected_len => {
            Ok(array.data.iter().map(|s| Some(s.clone())).collect())
        }
        Value::CharArray(chars) if chars.rows == expected_len => Ok((0..chars.rows)
            .map(|row| {
                let start = row * chars.cols;
                let end = start + chars.cols;
                Some(chars.data[start..end].iter().collect::<String>().trim().to_string())
            })
            .collect()),
        Value::Cell(cell) => group_labels_from_cell(cell, expected_len),
        Value::Object(object) if object.is_class("categorical") => (0..expected_len)
            .map(|row| {
                crate::builtins::table::categorical_label_at(object, row)
                    .map(Some)
                    .ok_or_else(|| invalid("scatterhist: invalid categorical Group value"))
            })
            .collect(),
        _ => Err(invalid(
            "scatterhist: Group must have one numeric, text, cell, or categorical label per observation",
        )),
    }
}

fn group_labels_from_cell(
    cell: &CellArray,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<String>>> {
    if cell.data.len() != expected_len {
        return Err(invalid(
            "scatterhist: Group must have one label per observation",
        ));
    }
    cell.data
        .iter()
        .map(|value| match value {
            Value::Num(value) => Ok(value.is_finite().then(|| number_label(*value))),
            Value::Int(value) => Ok(Some(integer_label(value))),
            Value::String(value) => Ok(Some(value.clone())),
            Value::CharArray(chars) => Ok(Some(chars.data.iter().collect())),
            Value::StringArray(array) if array.data.len() == 1 => Ok(Some(array.data[0].clone())),
            _ => Err(invalid("scatterhist: cell Group labels must be scalar")),
        })
        .collect()
}

fn number_label(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn integer_label(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

fn integer_values(tensor: &Tensor) -> Option<Vec<IntValue>> {
    tensor
        .integer_storage()
        .map(|storage| storage.exact_values())
}

fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

fn grouped_series(
    x: &[f64],
    y: &[f64],
    options: &ScatterhistOptions,
) -> BuiltinResult<Vec<GroupSeries>> {
    let labels = if options.plot_group {
        options
            .group
            .clone()
            .unwrap_or_else(|| vec![Some(String::new()); x.len()])
    } else {
        vec![Some(String::new()); x.len()]
    };
    let mut grouped: Vec<(String, GroupSeries)> = Vec::new();
    for ((&x_value, &y_value), label) in x.iter().zip(y.iter()).zip(labels.iter()) {
        let Some(label) = label else {
            continue;
        };
        let key = if label.is_empty() {
            "Data".to_string()
        } else {
            label.clone()
        };
        let idx = if let Some(idx) = grouped.iter().position(|(existing, _)| existing == &key) {
            idx
        } else {
            if grouped.len() >= MAX_GROUPS {
                return Err(invalid(format!(
                    "scatterhist: Group creates more than {MAX_GROUPS} groups"
                )));
            }
            let color_index = grouped.len();
            let style = series_style_for_group(color_index, options);
            grouped.push((
                key.clone(),
                GroupSeries {
                    label: key.clone(),
                    scatter_x: Vec::new(),
                    scatter_y: Vec::new(),
                    marginal_x: Vec::new(),
                    marginal_y: Vec::new(),
                    style,
                },
            ));
            grouped.len() - 1
        };
        let series = &mut grouped[idx].1;
        if x_value.is_finite() {
            series.marginal_x.push(x_value);
        }
        if y_value.is_finite() {
            series.marginal_y.push(y_value);
        }
        if x_value.is_finite() && y_value.is_finite() {
            series.scatter_x.push(x_value);
            series.scatter_y.push(y_value);
        }
    }
    if grouped.iter().all(|(_, group)| {
        group.scatter_x.is_empty() && group.marginal_x.is_empty() && group.marginal_y.is_empty()
    }) {
        return Err(invalid("scatterhist: no finite observations are available"));
    }
    Ok(grouped.into_iter().map(|(_, group)| group).collect())
}

fn series_style_for_group(index: usize, options: &ScatterhistOptions) -> SeriesStyle {
    let color = options
        .colors
        .as_ref()
        .and_then(|colors| colors.get(index % colors.len()).copied())
        .unwrap_or_else(|| GROUP_COLORS[index % GROUP_COLORS.len()]);
    SeriesStyle {
        color,
        marker: options.markers[index % options.markers.len()],
        marker_size: options.marker_sizes[index % options.marker_sizes.len()],
        line_width: options.line_widths[index % options.line_widths.len()],
        line_style: options.line_styles[index % options.line_styles.len()],
        bandwidth: if options.bandwidths.is_empty() {
            None
        } else {
            Some(options.bandwidths[index % options.bandwidths.len()])
        },
    }
}

fn render_scatterhist(
    series: &[GroupSeries],
    options: &ScatterhistOptions,
) -> BuiltinResult<[f64; 3]> {
    let figure = options
        .parent
        .unwrap_or_else(|| current_axes_state().handle);
    select_axes_for_figure(figure, 0).map_err(|err| internal(format!("scatterhist: {err}")))?;
    let layout = axes_for_layout(options.location, options.direction);

    for axes in [layout.empty, layout.xhist, layout.scatter, layout.yhist] {
        configure_subplot(2, 2, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
        select_axes_for_figure(figure, axes)
            .map_err(|err| internal(format!("scatterhist: {err}")))?;
        handle_plot_render_result(render_active_plot(
            NAME,
            PlotRenderOptions::default(),
            |_figure, _axes| Ok(()),
        ))?;
    }

    let x_samples = series
        .iter()
        .flat_map(|s| s.marginal_x.iter().copied())
        .collect::<Vec<_>>();
    let y_samples = series
        .iter()
        .flat_map(|s| s.marginal_y.iter().copied())
        .collect::<Vec<_>>();
    let x_limits = combined_limits(x_samples.iter().copied());
    let y_limits = combined_limits(y_samples.iter().copied());
    let bins = options.bins.unwrap_or([
        auto_bin_count(&x_samples, x_limits),
        auto_bin_count(&y_samples, y_limits),
    ]);
    let x_edges = bin_edges(x_limits, bins[0]);
    let y_edges = bin_edges(y_limits, bins[1]);

    render_scatter_axes(figure, layout.scatter, series)?;
    render_x_marginal(
        figure,
        layout.xhist,
        series,
        options,
        &x_edges,
        layout.x_marginal_scale,
    )?;
    render_y_marginal(
        figure,
        layout.yhist,
        series,
        options,
        &y_edges,
        layout.y_marginal_scale,
    )?;

    if options
        .legend
        .unwrap_or(options.group.is_some() && options.plot_group)
    {
        set_legend_for_axes(figure, layout.scatter, true, None, None)
            .map_err(|err| internal(format!("scatterhist: {err}")))?;
    }

    Ok([
        encode_axes_handle(figure, layout.scatter),
        encode_axes_handle(figure, layout.xhist),
        encode_axes_handle(figure, layout.yhist),
    ])
}

fn axes_for_layout(location: Location, direction: Direction) -> LayoutAxes {
    let (scatter, xhist, yhist, empty, xhist_is_south, yhist_is_west) = match location {
        // Location names the marginal histogram corner. For example, SouthWest
        // places the x histogram below and y histogram left of the scatter axes.
        Location::NorthEast => (2, 0, 3, 1, false, false),
        Location::SouthEast => (0, 2, 1, 3, true, false),
        Location::SouthWest => (1, 3, 0, 2, true, true),
        Location::NorthWest => (3, 1, 2, 0, false, true),
    };
    let x_marginal_scale = directional_scale(direction, xhist_is_south);
    let y_marginal_scale = directional_scale(direction, yhist_is_west);
    LayoutAxes {
        scatter,
        xhist,
        yhist,
        empty,
        x_marginal_scale,
        y_marginal_scale,
    }
}

fn directional_scale(direction: Direction, positive_points_toward_scatter: bool) -> f64 {
    match (direction, positive_points_toward_scatter) {
        (Direction::In, true) | (Direction::Out, false) => 1.0,
        (Direction::In, false) | (Direction::Out, true) => -1.0,
    }
}

fn render_scatter_axes(
    figure: crate::builtins::plotting::FigureHandle,
    axes: usize,
    series: &[GroupSeries],
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "Scatterhist",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        |figure, axes| {
            for item in series {
                if item.scatter_x.is_empty() {
                    continue;
                }
                let mut plot = ScatterPlot::new(item.scatter_x.clone(), item.scatter_y.clone())
                    .map_err(|err| internal(format!("scatterhist: {err}")))?;
                plot.color = item.style.color;
                plot.edge_color = item.style.color;
                plot.marker_size = item.style.marker_size;
                plot.marker_style = item.style.marker;
                plot.label = Some(item.label.clone());
                figure.add_scatter_plot_on_axes(plot, axes);
            }
            Ok(())
        },
    ))
}

fn render_x_marginal(
    figure: crate::builtins::plotting::FigureHandle,
    axes: usize,
    series: &[GroupSeries],
    options: &ScatterhistOptions,
    edges: &[f64],
    marginal_scale: f64,
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    let hist_style = resolved_hist_style(options, series);
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "X Marginal",
            x_label: "X",
            y_label: "Count",
            ..Default::default()
        },
        |figure, axes| {
            for (idx, item) in series.iter().enumerate() {
                if matches!(options.kernel, KernelMode::On | KernelMode::Overlay) {
                    let (xs, ys) =
                        kde_line(&item.marginal_x, edges, item.style.bandwidth.map(|b| b[0]));
                    let ys = scale_values(ys, marginal_scale);
                    let line = LinePlot::new(xs, ys)
                        .map_err(|err| internal(format!("scatterhist: {err}")))?
                        .with_style(
                            item.style.color,
                            item.style.line_width,
                            item.style.line_style,
                        )
                        .with_label(item.label.clone());
                    figure.add_line_plot_on_axes(line, axes);
                }
                if matches!(options.kernel, KernelMode::Off | KernelMode::Overlay) {
                    let counts =
                        scale_values(histogram_counts(&item.marginal_x, edges), marginal_scale);
                    match hist_style {
                        HistStyle::Bar => {
                            let mut bar =
                                bar_from_counts(edges, counts, item.style.color, &item.label)?;
                            bar = bar.with_group(idx, series.len());
                            figure.add_bar_chart_on_axes(bar, axes);
                        }
                        HistStyle::Stairs => {
                            let (xs, ys) = stairs_from_counts(edges, &counts, false);
                            let line = LinePlot::new(xs, ys)
                                .map_err(|err| internal(format!("scatterhist: {err}")))?
                                .with_style(
                                    item.style.color,
                                    item.style.line_width,
                                    item.style.line_style,
                                )
                                .with_label(item.label.clone());
                            figure.add_line_plot_on_axes(line, axes);
                        }
                    }
                }
            }
            Ok(())
        },
    ))
}

fn render_y_marginal(
    figure: crate::builtins::plotting::FigureHandle,
    axes: usize,
    series: &[GroupSeries],
    options: &ScatterhistOptions,
    edges: &[f64],
    marginal_scale: f64,
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    let hist_style = resolved_hist_style(options, series);
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "Y Marginal",
            x_label: "Count",
            y_label: "Y",
            ..Default::default()
        },
        |figure, axes| {
            for (idx, item) in series.iter().enumerate() {
                if matches!(options.kernel, KernelMode::On | KernelMode::Overlay) {
                    let (density, ys) =
                        kde_line(&item.marginal_y, edges, item.style.bandwidth.map(|b| b[1]));
                    let density = scale_values(density, marginal_scale);
                    let line = LinePlot::new(density, ys)
                        .map_err(|err| internal(format!("scatterhist: {err}")))?
                        .with_style(
                            item.style.color,
                            item.style.line_width,
                            item.style.line_style,
                        )
                        .with_label(item.label.clone());
                    figure.add_line_plot_on_axes(line, axes);
                }
                if matches!(options.kernel, KernelMode::Off | KernelMode::Overlay) {
                    let counts =
                        scale_values(histogram_counts(&item.marginal_y, edges), marginal_scale);
                    match hist_style {
                        HistStyle::Bar => {
                            let mut bar =
                                bar_from_counts(edges, counts, item.style.color, &item.label)?
                                    .with_orientation(Orientation::Horizontal);
                            bar = bar.with_group(idx, series.len());
                            figure.add_bar_chart_on_axes(bar, axes);
                        }
                        HistStyle::Stairs => {
                            let (xs, ys) = stairs_from_counts(edges, &counts, true);
                            let line = LinePlot::new(xs, ys)
                                .map_err(|err| internal(format!("scatterhist: {err}")))?
                                .with_style(
                                    item.style.color,
                                    item.style.line_width,
                                    item.style.line_style,
                                )
                                .with_label(item.label.clone());
                            figure.add_line_plot_on_axes(line, axes);
                        }
                    }
                }
            }
            Ok(())
        },
    ))
}

fn bar_from_counts(
    edges: &[f64],
    counts: Vec<f64>,
    color: Vec4,
    label: &str,
) -> BuiltinResult<BarChart> {
    let labels = edges
        .windows(2)
        .map(|pair| format!("[{:.3}, {:.3})", pair[0], pair[1]))
        .collect::<Vec<_>>();
    let mut chart = BarChart::new(labels, counts)
        .map_err(|err| internal(format!("scatterhist: {err}")))?
        .with_style(color, 0.95)
        .with_label(label.to_string());
    chart.set_histogram_bin_edges(edges.to_vec());
    Ok(chart)
}

fn resolved_hist_style(options: &ScatterhistOptions, series: &[GroupSeries]) -> HistStyle {
    options.hist_style.unwrap_or_else(|| {
        if options.group.is_some() && options.plot_group && series.len() > 1 {
            HistStyle::Stairs
        } else {
            HistStyle::Bar
        }
    })
}

fn handle_plot_render_result(result: BuiltinResult<String>) -> BuiltinResult<()> {
    match result {
        Ok(_) => Ok(()),
        Err(err) => {
            let lower = err.to_string().to_lowercase();
            if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
                Ok(())
            } else {
                Err(internal(err.message))
            }
        }
    }
}

fn scale_values(mut values: Vec<f64>, scale: f64) -> Vec<f64> {
    if scale != 1.0 {
        for value in &mut values {
            *value *= scale;
        }
    }
    values
}

fn stairs_from_counts(edges: &[f64], counts: &[f64], horizontal: bool) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(counts.len() * 2);
    let mut y = Vec::with_capacity(counts.len() * 2);
    for (idx, count) in counts.iter().copied().enumerate() {
        let left = edges[idx];
        let right = edges[idx + 1];
        if horizontal {
            x.push(count);
            y.push(left);
            x.push(count);
            y.push(right);
        } else {
            x.push(left);
            y.push(count);
            x.push(right);
            y.push(count);
        }
    }
    (x, y)
}

fn combined_limits(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for value in values.filter(|value| value.is_finite()) {
        lo = Some(lo.map_or(value, |current| current.min(value)));
        hi = Some(hi.map_or(value, |current| current.max(value)));
    }
    let (mut lo, mut hi) = match (lo, hi) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (0.0, 1.0),
    };
    if lo == hi {
        let pad = lo.abs().max(1.0) * 0.5;
        lo -= pad;
        hi += pad;
    }
    (lo, hi)
}

fn bin_edges((lo, hi): (f64, f64), bins: usize) -> Vec<f64> {
    let step = (hi - lo) / bins as f64;
    let mut edges = (0..=bins)
        .map(|idx| lo + step * idx as f64)
        .collect::<Vec<_>>();
    if let Some(last) = edges.last_mut() {
        *last = hi;
    }
    edges
}

fn auto_bin_count(data: &[f64], limits: (f64, f64)) -> usize {
    let finite = data
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value >= limits.0 && *value <= limits.1)
        .collect::<Vec<_>>();
    if finite.len() < 2 {
        return 2;
    }
    let width = scott_bin_width(&finite);
    if !width.is_finite() || width <= 0.0 {
        return 2;
    }
    let bins = ((limits.1 - limits.0) / width).ceil().max(2.0) as usize;
    bins.clamp(2, DEFAULT_AUTO_BIN_CAP.min(MAX_BINS))
}

fn scott_bin_width(data: &[f64]) -> f64 {
    let sigma = sample_std(data);
    if sigma > 0.0 {
        3.5 * sigma / (data.len() as f64).powf(1.0 / 3.0)
    } else {
        0.0
    }
}

fn histogram_counts(data: &[f64], edges: &[f64]) -> Vec<f64> {
    let bins = edges.len().saturating_sub(1);
    let mut counts = vec![0.0; bins];
    for value in data.iter().copied().filter(|value| value.is_finite()) {
        if value < edges[0] || value > edges[bins] {
            continue;
        }
        let idx = if value == edges[bins] {
            bins - 1
        } else {
            edges
                .partition_point(|edge| *edge <= value)
                .saturating_sub(1)
        };
        if let Some(count) = counts.get_mut(idx) {
            *count += 1.0;
        }
    }
    counts
}

fn kde_line(data: &[f64], edges: &[f64], bandwidth: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    let centers = edges
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect::<Vec<_>>();
    let finite = data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (centers, vec![0.0; edges.len().saturating_sub(1)]);
    }
    let bandwidth = bandwidth
        .unwrap_or_else(|| scott_bandwidth(&finite).max((edges[1] - edges[0]).abs() / 2.0));
    let norm = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * bandwidth);
    let mut values = Vec::with_capacity(centers.len());
    for center in &centers {
        let density = finite
            .iter()
            .map(|value| {
                let z = (*center - *value) / bandwidth;
                (-0.5 * z * z).exp() * norm
            })
            .sum::<f64>();
        values.push(density * finite.len() as f64);
    }
    (centers, values)
}

fn scott_bandwidth(data: &[f64]) -> f64 {
    let sigma = sample_std(data);
    if sigma > 0.0 {
        1.06 * sigma * (data.len() as f64).powf(-0.2)
    } else {
        1.0
    }
}

fn sample_std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 1.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (data.len() as f64 - 1.0);
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::{current_figure_handle, decode_axes_handle};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clone_figure, reset_hold_state_for_run};
    use runmat_builtins::{CellArray, IntegerStorage, StringArray};
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn first_unrepresentable_usize_double() -> f64 {
        if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        }
    }

    #[test]
    fn scatterhist_numeric_parsers_read_typed_integer_storage_exactly() {
        let wide = u64::MAX - 1;
        assert_eq!(
            numeric_vector(
                int_tensor(IntegerStorage::U64(vec![wide, wide - 1]), vec![2, 1]),
                "x",
            )
            .unwrap(),
            vec![
                IntValue::U64(wide).to_f64(),
                IntValue::U64(wide - 1).to_f64()
            ]
        );
        assert_eq!(
            parse_bins(&int_tensor(IntegerStorage::U16(vec![12, 8]), vec![1, 2])).unwrap(),
            [12, 8]
        );
        assert!(option_bool(
            &int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            "Legend",
        )
        .unwrap());
        assert_eq!(
            nonnegative_scalar_cycle(
                &int_tensor(IntegerStorage::U16(vec![4, 5]), vec![1, 2]),
                "MarkerSize",
            )
            .unwrap(),
            vec![4.0, 5.0]
        );
        assert_eq!(
            group_labels(
                &int_tensor(IntegerStorage::U64(vec![wide, wide - 1]), vec![2, 1]),
                2,
            )
            .unwrap(),
            vec![Some(wide.to_string()), Some((wide - 1).to_string())]
        );
    }

    #[test]
    fn scatterhist_rejects_invalid_typed_integer_options() {
        let err = parse_bins(&int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1])).unwrap_err();
        assert!(
            err.message.contains("greater than or equal to 2"),
            "{}",
            err.message
        );

        let err = option_bool(
            &int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            "Legend",
        )
        .unwrap_err();
        assert!(err.message.contains("0 or 1"), "{}", err.message);
    }

    #[test]
    fn scatterhist_rejects_unrepresentable_double_bin_counts_before_casting() {
        let err = parse_bins(&vec_tensor(&[first_unrepresentable_usize_double()])).unwrap_err();
        assert!(
            err.message.contains("no greater than 250"),
            "{}",
            err.message
        );
    }

    #[test]
    fn scatterhist_accepts_typed_integer_data_and_options() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();

        let _ = futures::executor::block_on(scatterhist_builtin(vec![
            int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
            int_tensor(IntegerStorage::I16(vec![3, 2, 1, 0]), vec![4, 1]),
            Value::String("NBins".into()),
            int_tensor(IntegerStorage::U16(vec![2, 2]), vec![1, 2]),
            Value::String("Group".into()),
            int_tensor(IntegerStorage::U8(vec![1, 2, 1, 2]), vec![4, 1]),
            Value::String("MarkerSize".into()),
            int_tensor(IntegerStorage::U16(vec![4, 5]), vec![1, 2]),
            Value::String("LineWidth".into()),
            int_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
            Value::String("Color".into()),
            int_tensor(IntegerStorage::U8(vec![1, 0, 0, 0, 1, 0]), vec![2, 3]),
            Value::String("Legend".into()),
            int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
        ]))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        let scatter_plots = figure
            .plots()
            .filter_map(|plot| match plot {
                PlotElement::Scatter(scatter) => Some(scatter),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scatter_plots.len(), 2);
        assert_eq!(scatter_plots[0].marker_size, 4.0);
        assert_eq!(scatter_plots[1].marker_size, 5.0);
        assert!(figure.axes_metadata(1).unwrap().legend_enabled);
    }

    #[test]
    fn scatterhist_renders_three_axes_with_histograms() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();

        let handles = futures::executor::block_on(scatterhist_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::String("NBins".into()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let Value::Tensor(handles) = handles else {
            panic!("expected handle tensor");
        };
        assert_eq!(handles.shape, vec![3, 1]);
        let decoded = handles
            .materialize_f64()
            .iter()
            .map(|handle| decode_axes_handle(*handle).unwrap().1)
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![1, 3, 0]);

        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.axes_grid(), (2, 2));
        assert_eq!(figure.len(), 3);
        assert!(matches!(
            figure.plots().next().unwrap(),
            PlotElement::Scatter(_)
        ));
        let bars = figure
            .plots()
            .filter(|plot| matches!(plot, PlotElement::Bar(_)))
            .count();
        assert_eq!(bars, 2);
        let axes = figure.plot_axes_indices();
        assert_eq!(axes, &[1, 3, 0]);
    }

    #[test]
    fn scatterhist_groups_and_kernel_lines() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();

        let groups = StringArray::new(
            vec!["a".into(), "b".into(), "a".into(), "b".into()],
            vec![4, 1],
        )
        .unwrap();
        let _ = futures::executor::block_on(scatterhist_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::String("Group".into()),
            Value::StringArray(groups),
            Value::String("Kernel".into()),
            Value::String("on".into()),
        ]))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        let scatter_count = figure
            .plots()
            .filter(|plot| matches!(plot, PlotElement::Scatter(_)))
            .count();
        let line_count = figure
            .plots()
            .filter(|plot| matches!(plot, PlotElement::Line(_)))
            .count();
        assert_eq!(scatter_count, 2);
        assert_eq!(line_count, 4);
        assert!(figure.axes_metadata(1).unwrap().legend_enabled);
    }

    #[test]
    fn scatterhist_grouped_defaults_to_stairs_and_cycles_styles() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();

        let groups = StringArray::new(
            vec!["a".into(), "b".into(), "a".into(), "b".into()],
            vec![4, 1],
        )
        .unwrap();
        let line_styles = CellArray::new(
            vec![Value::String("-".into()), Value::String("--".into())],
            1,
            2,
        )
        .unwrap();
        let _ = futures::executor::block_on(scatterhist_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::String("Group".into()),
            Value::StringArray(groups),
            Value::String("Color".into()),
            Value::String("kr".into()),
            Value::String("Marker".into()),
            Value::String("o+".into()),
            Value::String("MarkerSize".into()),
            Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap()),
            Value::String("LineStyle".into()),
            Value::Cell(line_styles),
            Value::String("LineWidth".into()),
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        let scatter_plots = figure
            .plots()
            .filter_map(|plot| match plot {
                PlotElement::Scatter(scatter) => Some(scatter),
                _ => None,
            })
            .collect::<Vec<_>>();
        let line_plots = figure
            .plots()
            .filter_map(|plot| match plot {
                PlotElement::Line(line) => Some(line),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scatter_plots.len(), 2);
        assert_eq!(line_plots.len(), 4);
        assert_eq!(
            figure
                .plots()
                .filter(|plot| matches!(plot, PlotElement::Bar(_)))
                .count(),
            0
        );
        assert_eq!(scatter_plots[0].color, Vec4::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(scatter_plots[0].marker_style, MarkerStyle::Circle);
        assert_eq!(scatter_plots[0].marker_size, 4.0);
        assert_eq!(scatter_plots[1].color, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(scatter_plots[1].marker_style, MarkerStyle::Plus);
        assert_eq!(scatter_plots[1].marker_size, 5.0);
        assert_eq!(line_plots[0].line_style, LineStyle::Solid);
        assert_eq!(line_plots[0].line_width, 2.0);
        assert_eq!(line_plots[1].line_style, LineStyle::Dashed);
        assert_eq!(line_plots[1].line_width, 3.0);
    }

    #[test]
    fn scatterhist_keeps_marginal_nan_semantics_independent() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();

        let _ = futures::executor::block_on(scatterhist_builtin(vec![
            vec_tensor(&[1.0, 2.0, f64::NAN]),
            vec_tensor(&[10.0, f64::NAN, 30.0]),
            Value::String("NBins".into()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        let mut scatter_points = 0usize;
        let mut x_count = 0.0;
        let mut y_count = 0.0;
        for (plot, axes) in figure.plots().zip(figure.plot_axes_indices()) {
            match (plot, *axes) {
                (PlotElement::Scatter(scatter), 1) => {
                    scatter_points += scatter.host_xy_f64().unwrap().unwrap().0.len()
                }
                (PlotElement::Bar(bar), 3) => {
                    x_count = bar.values().unwrap().iter().map(|v| v.abs()).sum()
                }
                (PlotElement::Bar(bar), 0) => {
                    y_count = bar.values().unwrap().iter().map(|v| v.abs()).sum()
                }
                _ => {}
            }
        }
        assert_eq!(scatter_points, 1);
        assert_eq!(x_count, 2.0);
        assert_eq!(y_count, 2.0);
    }

    #[test]
    fn scatterhist_rejects_invalid_bins_and_supports_stairs_overlay_options() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();

        let err = futures::executor::block_on(scatterhist_builtin(vec![
            vec_tensor(&[0.0, 1.0]),
            vec_tensor(&[0.0, 1.0]),
            Value::String("NBins".into()),
            Value::Num(1.0),
        ]))
        .unwrap_err();
        assert!(err.message.contains("greater than or equal to 2"));

        let err = futures::executor::block_on(scatterhist_builtin(vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap()),
            vec_tensor(&[0.0, 1.0, 2.0, 3.0]),
        ]))
        .unwrap_err();
        assert!(err.message.contains("x must be a vector"));

        let err = futures::executor::block_on(scatterhist_builtin(vec![
            vec_tensor(&[0.0, 1.0]),
            vec_tensor(&[0.0, 1.0]),
            Value::String("Style".into()),
            Value::String("overlay".into()),
        ]))
        .unwrap_err();
        assert!(err.message.contains("unsupported Style"));

        let _ = futures::executor::block_on(scatterhist_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::String("Style".into()),
            Value::String("stairs".into()),
            Value::String("LineStyle".into()),
            Value::String("--".into()),
            Value::String("MarkerSize".into()),
            Value::Num(0.0),
        ]))
        .unwrap();
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            figure
                .plots()
                .filter(|plot| matches!(plot, PlotElement::Line(_)))
                .count(),
            2
        );
    }
}
