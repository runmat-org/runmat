//! Scatter plot with marginal histograms.
mod admission;
mod controls;
mod grouping;
mod labels;
mod marginal;
mod numeric;
mod plan;
mod render;
mod styles;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod provider_tests;
use admission::*;
use controls::*;
use grouping::*;
use labels::*;
use marginal::*;
use numeric::*;
use plan::*;
use render::*;
use styles::*;

use glam::Vec4;
use runmat_builtins::{
    BuiltinErrorDescriptor, ScatterhistIntegerRole, ScatterhistOptionKind,
    SCATTERHIST_ERROR_INTERNAL, SCATTERHIST_ERROR_INVALID_ARGUMENT,
    SCATTERHIST_GPU_INPUT_EXTENSION, SCATTERHIST_INTEGER_DATA_EXTENSION,
    SCATTERHIST_INTEGER_GROUP_EXTENSION, SCATTERHIST_INTEGER_OPTION_EXTENSION,
    SCATTERHIST_MAX_BINS,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{
    bar::Orientation, BarChart, LinePlot, LineStyle, MarkerStyle, ScatterPlot,
};
use runmat_value::{CellArray, IntValue, Tensor, Value};

use crate::builtins::common::grouping_keys::KeyAtom;
use crate::builtins::common::tensor;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    configure_subplot, current_axes_state, encode_axes_handle, render_active_plot,
    select_axes_for_figure, set_legend_for_axes, FigureHandle, PlotRenderOptions,
};
use crate::builtins::plotting::style::{parse_color_value, LineStyleParseOptions};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "scatterhist";
const MAX_BINS: usize = SCATTERHIST_MAX_BINS;
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

#[derive(Clone, Debug)]
struct ScatterhistOptions {
    bins: Option<[usize; 2]>,
    group: Option<Vec<Option<KeyAtom>>>,
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

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&SCATTERHIST_ERROR_INVALID_ARGUMENT, message)
}

fn invalid_with_source(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    let span = source.span;
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_gpu_gather_retry(source.gpu_gather_retry())
        .with_source(source);
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    if let Some(identifier) = SCATTERHIST_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&SCATTERHIST_ERROR_INTERNAL, message)
}

#[cfg(test)]
pub(crate) async fn scatterhist_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args =
        crate::execution::RootedValues::new(args).map_err(|error| error.into_runtime_error())?;
    scatterhist_builtin_registered(None, args.into()).await
}

#[runtime_builtin(
    name = "scatterhist",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::scatterhist"
)]
pub(crate) async fn scatterhist_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(invalid("scatterhist: expected x and y inputs"));
    }
    let options = args.range(2..)?;
    let _option_sources = options.fork();
    let data = args.into_range(..2)?;
    admit_data_compatibility(data.view())?;
    let observations = admit_host_visible(data.view())?;
    let option_plan = ScatterhistOptionPlan::admit(options, observations)?;
    let data = gather_values(data, runtime.as_ref()).await?;
    let (x, y) = parse_xy(data.view())?;
    let options = option_plan.evaluate(x.len()).await?;
    let series = grouped_series(&x, &y, &options)?;
    let axes = render_scatterhist(&series, &options)?;
    Ok(Value::Tensor(
        Tensor::new(axes.to_vec(), vec![3, 1])
            .map_err(|err| internal(format!("scatterhist: {err}")))?,
    ))
}
