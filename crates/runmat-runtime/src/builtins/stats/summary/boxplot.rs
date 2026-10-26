//! Box plot compatibility helper.

use std::collections::BTreeMap;

use glam::Vec4;
use runmat_builtins::{
    BoxplotOptionKind, BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BOXPLOT_ERROR_INTERNAL,
    BOXPLOT_ERROR_INVALID_ARGUMENT, BOXPLOT_GPU_INPUT_EXTENSION, BOXPLOT_INTEGER_DATA_EXTENSION,
    BOXPLOT_INTEGER_GROUP_EXTENSION, BOXPLOT_INTEGER_OPTION_EXTENSION,
    BOXPLOT_LOGICAL_DATA_EXTENSION, BOXPLOT_LOGICAL_GROUP_EXTENSION,
    BOXPLOT_LOGICAL_OPTION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{LinePlot, MarkerStyle, PatchPlot, ScatterPlot};
use runmat_value::{CellArray, CharArray, NumericScalar, Tensor, Value};

use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_arguments};
use crate::builtins::plotting::state::{render_active_plot, PlotRenderOptions};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "boxplot";
mod admission;
mod extension_scan;
mod appearance;
mod cooperation;
mod factor_layout;
mod plan;
use crate::builtins::common::grouping_keys::KeyAtom;
use appearance::{BoxStyle, LabelOrientation, LabelVerbosity, MedianStyle, NotchStyle};
use factor_layout::{FactorDirection, FactorGap, FactorSeparator};
use plan::BoxplotOptionPlan;

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
    box_style: BoxStyle,
    box_style_explicit: bool,
    median_style: MedianStyle,
    median_style_explicit: bool,
    notch_style: NotchStyle,
    label_orientation: LabelOrientation,
    label_orientation_explicit: bool,
    label_verbosity: LabelVerbosity,
    label_verbosity_explicit: bool,
    factor_direction: Vec<FactorDirection>,
    full_factors: bool,
    factor_gap: FactorGap,
    factor_gap_explicit: bool,
    factor_separator: FactorSeparator,
    factor_separator_explicit: bool,
}

#[derive(Clone, Debug)]
struct BoxStats {
    label: String,
    group_key: Vec<KeyAtom>,
    position: f64,
    width: f64,
    color: Vec4,
    q1: f64,
    median: f64,
    q3: f64,
    lower_whisker: f64,
    upper_whisker: f64,
    outliers: Vec<f64>,
    observation_count: usize,
}

#[derive(Clone, Debug)]
enum PlotPlan {
    Line(LinePlot),
    Scatter(ScatterPlot),
    Patch(PatchPlot),
}

#[derive(Clone, Copy, Debug)]
enum HandleKind {
    Line,
    Scatter,
    Patch,
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
            box_style: BoxStyle::Outline,
            box_style_explicit: false,
            median_style: MedianStyle::Line,
            median_style_explicit: false,
            notch_style: NotchStyle::Off,
            label_orientation: LabelOrientation::Horizontal,
            label_orientation_explicit: false,
            label_verbosity: LabelVerbosity::All,
            label_verbosity_explicit: false,
            factor_direction: vec![FactorDirection::Data],
            full_factors: false,
            factor_gap: FactorGap::None,
            factor_gap_explicit: false,
            factor_separator: FactorSeparator::None,
            factor_separator_explicit: false,
        }
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
    error(&BOXPLOT_ERROR_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&BOXPLOT_ERROR_INTERNAL, message)
}

fn allocation_failure(
    message: impl Into<String>,
    cause: std::collections::TryReserveError,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_source(cause);
    if let Some(identifier) = BOXPLOT_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_with_source(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    with_owned_source(&BOXPLOT_ERROR_INVALID_ARGUMENT, message, source)
}

fn internal_with_source(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    with_owned_source(&BOXPLOT_ERROR_INTERNAL, message, source)
}

fn with_owned_source(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: RuntimeError,
) -> RuntimeError {
    let span = source.span;
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_gpu_gather_retry(source.gpu_gather_retry())
        .with_source(source);
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_plot_error(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        crate::builtins::plotting::state::with_render_cause(invalid(err.message.clone()), err)
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

fn value_contains_gpu(value: &Value) -> BuiltinResult<bool> {
    let mut found = false;
    crate::dispatcher::try_visit_gpu_handles(value, &mut |_| found = true)
        .map_err(|err| invalid_with_source(format!("boxplot: {err}"), err))?;
    Ok(found)
}

fn enable_boxplot_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)
}

async fn ensure_sample_group_extensions(
    x: &Value,
    group: Option<&Value>,
    cooperation: &mut cooperation::BoxplotCooperation,
) -> BuiltinResult<()> {
    cooperation.completed(0).await?;
    if value_is_integer_numeric(x) {
        enable_boxplot_extension(&BOXPLOT_INTEGER_DATA_EXTENSION)?;
    }
    if value_is_logical_numeric(x) {
        enable_boxplot_extension(&BOXPLOT_LOGICAL_DATA_EXTENSION)?;
    }
    if let Some(group) = group {
        let roles = extension_scan::numeric_roles_cooperative(group, cooperation).await?;
        if roles.integer {
            enable_boxplot_extension(&BOXPLOT_INTEGER_GROUP_EXTENSION)?;
        }
        if roles.logical {
            enable_boxplot_extension(&BOXPLOT_LOGICAL_GROUP_EXTENSION)?;
        }
    }
    let group_contains_gpu = match group {
        Some(group) => value_contains_gpu(group)?,
        None => false,
    };
    if value_contains_gpu(x)? || group_contains_gpu {
        enable_boxplot_extension(&BOXPLOT_GPU_INPUT_EXTENSION)?;
    }
    Ok(())
}

#[runtime_builtin(
    name = "boxplot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::boxplot"
)]
pub(crate) async fn boxplot_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    boxplot_in_runtime(runtime.as_ref(), args).await
}

#[cfg(test)]
pub(crate) async fn boxplot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    boxplot_in_runtime(
        None,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}

pub(super) async fn boxplot_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = cooperation::BoxplotCooperation::new(runtime);
    let (axes_target, args) = split_leading_axes_arguments(args, NAME).map_err(map_plot_error)?;
    if args.is_empty() {
        return Err(invalid("boxplot: expected numeric data"));
    }
    let _argument_sources = args.fork();
    let mut args = args;
    let x = args.pop_front().expect("data input checked above");
    let first_option = args
        .first()
        .map(keyword_of)
        .transpose()?
        .flatten()
        .and_then(|name| BoxplotOptionKind::parse(&name));
    let group = if !args.is_empty() && first_option.is_none() {
        Some(args.pop_front().expect("group input checked above"))
    } else {
        None
    };
    ensure_sample_group_extensions(&x, group.as_ref(), &mut cooperation).await?;
    admission::admit_host_visible(&x, group.as_ref(), &mut cooperation).await?;
    let option_plan = BoxplotOptionPlan::admit(args, first_option)?;
    option_plan
        .admit_host_color_groups(&x, &mut cooperation)
        .await?;
    option_plan.preflight_known_factor_count(admission::known_factor_count(group.as_ref()))?;
    cooperation.completed(0).await?;
    let x = gather_value(x).await?;
    let group = match group {
        Some(group) => Some(gather_value(group).await?),
        None => None,
    };
    let options = option_plan.evaluate().await?;
    let layout = groups::build_box_layout_cooperative(x, group, &options, &mut cooperation).await?;

    let prepared =
        plot::prepare_box_layout_cooperative(&layout, &options, &mut cooperation).await?;
    apply_axes_target(axes_target, NAME).map_err(map_plot_error)?;
    let handles = plot::publish_boxplot(prepared, &options)?;
    let value = handles_value(handles)?;

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![value],
        )),
        None => Ok(value),
    }
}

async fn gather_value(arg: Value) -> BuiltinResult<Value> {
    if value_contains_gpu(&arg)? {
        gather_if_needed_async(&arg)
            .await
            .map_err(|err| invalid_with_source(format!("boxplot: {err}"), err))
    } else {
        Ok(arg)
    }
}

mod plot;
use plot::handles_value;
#[cfg(test)]
use plot::render_boxplot;

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

mod options;

mod groups;
#[cfg(test)]
use groups::build_box_stats;
mod statistics;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod provider_tests;
