//! Reference-line compatibility helper.

#[cfg(test)]
use runmat_builtins::REFLINE_ERROR_INVALID_ARGUMENT;
use runmat_builtins::{
    BuiltinErrorDescriptor, ReferenceLineOwner, REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION,
    REFLINE_INTEGER_COEFFICIENT_EXTENSION, REFLINE_LOGICAL_COEFFICIENT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{LinePlot, LineStyle, PlotElement};
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::apply_axes_target;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    append_active_plot, axes_metadata_snapshot, clone_figure, current_axes_state,
    register_line_handle, FigureHandle, PlotRenderOptions,
};
use crate::builtins::plotting::style::{
    marker_metadata_from_appearance, parse_line_style_args, value_as_f64, LineStyleParseOptions,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "refline";
mod fit;
use fit::least_squares_coefficients;

trait ReferenceLineRuntimeErrors {
    fn invalid(self, message: impl Into<String>) -> RuntimeError;
    fn internal(self, message: impl Into<String>) -> RuntimeError;
    fn invalid_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError;
    fn internal_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError;
}

impl ReferenceLineRuntimeErrors for ReferenceLineOwner {
    fn invalid(self, message: impl Into<String>) -> RuntimeError {
        self.error(self.invalid_descriptor(), message)
    }

    fn internal(self, message: impl Into<String>) -> RuntimeError {
        self.error(self.internal_descriptor(), message)
    }

    fn invalid_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError {
        self.sourced_error(self.invalid_descriptor(), message, cause)
    }

    fn internal_with_source(self, message: impl Into<String>, cause: RuntimeError) -> RuntimeError {
        self.sourced_error(self.internal_descriptor(), message, cause)
    }
}

trait ReferenceLineErrorBuilder {
    fn error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
    ) -> RuntimeError;
    fn sourced_error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
        cause: RuntimeError,
    ) -> RuntimeError;
}

impl ReferenceLineErrorBuilder for ReferenceLineOwner {
    fn error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(message).with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    fn sourced_error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
        cause: RuntimeError,
    ) -> RuntimeError {
        let span = cause.span;
        let mut builder = build_runtime_error(message)
            .with_builtin(self.name())
            .with_gpu_gather_retry(cause.gpu_gather_retry())
            .with_source(cause);
        if let Some(span) = span {
            builder = builder.with_span(span);
        }
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

#[cfg(test)]
fn internal_error(message: impl Into<String>) -> RuntimeError {
    ReferenceLineOwner::Refline.internal(message)
}

#[runtime_builtin(
    name = "refline",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::refline"
)]
pub(crate) async fn refline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    run_reference_lines(args, ReferenceLineOwner::Refline).await
}

pub(super) async fn run_reference_lines(
    args: Vec<Value>,
    owner: ReferenceLineOwner,
) -> BuiltinResult<Value> {
    let (target, mut args) = split_optional_axes(args)?;
    let coefficient_count = coefficient_count(&args, owner)?;
    let style = parse_line_style_args(
        &args[coefficient_count..],
        &LineStyleParseOptions::generic(owner.name()),
    )
    .map_err(|err| {
        let message = err.message.clone();
        owner.invalid_with_source(message, err)
    })?;
    normalize_refline_coefficients(&mut args, owner, coefficient_count).await?;
    let (plan, style_args) = parse_refline_coefficients(&args, owner)?;
    debug_assert_eq!(style_args.len(), args.len() - coefficient_count);
    apply_axes_target(target, owner.name()).map_err(|err| {
        if matches!(owner, ReferenceLineOwner::Refline) && err.identifier().is_some() {
            return err;
        }
        let message = err.message.clone();
        owner.internal_with_source(message, err)
    })?;
    let axes = current_axes_state();
    let figure_handle = axes.handle;

    let specs = match plan {
        ReflinePlan::Explicit { slope, intercept } => vec![ReflineLineSpec {
            slope,
            intercept,
            x_span: refline_x_span(figure_handle, axes.active_index, owner)?,
        }],
        ReflinePlan::LeastSquares => {
            least_squares_specs(figure_handle, axes.active_index, owner).await?
        }
    };
    if specs.is_empty() {
        return Ok(Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).map_err(
            |err| owner.internal(format!("{}: {err}", owner.name())),
        )?));
    }

    let lines = specs
        .iter()
        .map(|spec| line_from_spec(*spec, &style, owner))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut lines = Some(lines);

    let plot_indices_slot = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_out = std::rc::Rc::clone(&plot_indices_slot);
    let render_result = append_active_plot(
        owner.name(),
        PlotRenderOptions::default(),
        move |figure, axes| {
            let lines = lines.take().ok_or_else(|| {
                owner.internal(format!("{}: lines already rendered", owner.name()))
            })?;
            let mut indices = plot_indices_out.borrow_mut();
            for line in lines {
                let index = figure.add_line_plot_on_axes(line, axes);
                indices.push((axes, index));
            }
            Ok(())
        },
    );

    if let Err(err) = render_result {
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Err(err);
        }
    }

    let handles = plot_indices_slot
        .borrow()
        .iter()
        .map(|(axes_index, plot_index)| {
            register_line_handle(figure_handle, *axes_index, *plot_index)
        })
        .collect::<Vec<_>>();
    if handles.is_empty() {
        return Err(owner.internal(format!("{}: line was not registered", owner.name())));
    }
    if handles.len() == 1 {
        return Ok(Value::Num(handles[0]));
    }

    let len = handles.len();
    Ok(Value::Tensor(Tensor::new(handles, vec![len, 1]).map_err(
        |err| owner.internal(format!("{}: {err}", owner.name())),
    )?))
}

fn coefficient_count(args: &[Value], owner: ReferenceLineOwner) -> BuiltinResult<usize> {
    if matches!(owner, ReferenceLineOwner::Lsline) {
        return if args.is_empty() {
            Ok(0)
        } else {
            Err(owner.invalid("lsline: expected zero inputs or one axes handle"))
        };
    }
    let Some(first) = args.first() else {
        return Ok(0);
    };
    let first_len = coefficient_element_count(first, owner)?;
    if matches!(first, Value::Tensor(_) | Value::GpuTensor(_)) && !matches!(first_len, 1 | 2) {
        return Err(owner.invalid("refline: coefficients must be scalar or a two-element vector"));
    }
    if first_len == 2 {
        Ok(1)
    } else if args.len() >= 2 {
        let second = &args[1];
        if matches!(second, Value::Tensor(_) | Value::GpuTensor(_))
            && coefficient_element_count(second, owner)? != 1
        {
            return Err(owner.invalid("refline: intercept must be scalar"));
        }
        Ok(2)
    } else {
        Err(owner
            .invalid("refline: expected coefficients as [slope intercept] or slope, intercept"))
    }
}

fn coefficient_element_count(value: &Value, owner: ReferenceLineOwner) -> BuiltinResult<usize> {
    if let Value::GpuTensor(handle) = value {
        if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
            return Err(owner.invalid("refline: coefficients must be real numeric values"));
        }
    }
    Ok(match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor),
        Value::GpuTensor(handle) => handle
            .shape()
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| {
                owner.invalid(format!("{}: coefficient shape overflows", owner.name()))
            })?,
        _ => 1,
    })
}

async fn normalize_refline_coefficients(
    args: &mut [Value],
    owner: ReferenceLineOwner,
    coefficient_count: usize,
) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_logical_class,
        value_has_native_integer_class,
    };
    for value in args.iter_mut().take(coefficient_count) {
        if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION,
                owner.name(),
            )?;
        }
        if value_has_logical_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &REFLINE_LOGICAL_COEFFICIENT_EXTENSION,
                owner.name(),
            )?;
        }
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &REFLINE_INTEGER_COEFFICIENT_EXTENSION,
                owner.name(),
            )?;
            if !native_integer_value_is_exact_f64_async(value).await? {
                return Err(owner.invalid(format!(
                    "{}: integer coefficients must be exactly representable as double",
                    owner.name()
                )));
            }
        }
        if matches!(value, Value::GpuTensor(_)) {
            *value = crate::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|err| {
                    let message = format!("{}: {err}", owner.name());
                    owner.invalid_with_source(message, err)
                })?;
        }
    }
    Ok(())
}

type AxesTarget = Option<(FigureHandle, usize)>;

#[derive(Clone, Copy)]
enum ReflinePlan {
    Explicit { slope: f64, intercept: f64 },
    LeastSquares,
}

#[derive(Clone, Copy)]
struct ReflineLineSpec {
    slope: f64,
    intercept: f64,
    x_span: (f64, f64),
}

fn split_optional_axes(args: Vec<Value>) -> BuiltinResult<(AxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, NAME) {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

fn parse_refline_coefficients(
    args: &[Value],
    owner: ReferenceLineOwner,
) -> BuiltinResult<(ReflinePlan, &[Value])> {
    match args {
        [] => Ok((ReflinePlan::LeastSquares, &[])),
        [coeffs, rest @ ..] => {
            if let Some(pair) = coefficient_pair(coeffs, owner)? {
                return Ok((
                    ReflinePlan::Explicit {
                        slope: pair.0,
                        intercept: pair.1,
                    },
                    rest,
                ));
            }
            if rest.is_empty() {
                return Err(owner.invalid(format!(
                    "{}: expected coefficients as [slope intercept] or slope, intercept",
                    owner.name()
                )));
            }
            let slope = finite_scalar(coeffs, "slope", owner)?;
            let intercept = finite_scalar(&rest[0], "intercept", owner)?;
            Ok((ReflinePlan::Explicit { slope, intercept }, &rest[1..]))
        }
    }
}

fn coefficient_pair(value: &Value, owner: ReferenceLineOwner) -> BuiltinResult<Option<(f64, f64)>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor_utils::tensor_element_len(tensor) != 2 {
                return Ok(None);
            }
            let slope = tensor_utils::tensor_value_f64(tensor, 0);
            let intercept = tensor_utils::tensor_value_f64(tensor, 1);
            if !slope.is_finite() || !intercept.is_finite() {
                return Err(owner.invalid(format!(
                    "{}: coefficients must contain finite slope and intercept",
                    owner.name()
                )));
            }
            Ok(Some((slope, intercept)))
        }
        _ => Ok(None),
    }
}

fn finite_scalar(value: &Value, name: &str, owner: ReferenceLineOwner) -> BuiltinResult<f64> {
    let value = value_as_f64(value)
        .ok_or_else(|| owner.invalid(format!("{}: {name} must be numeric", owner.name())))?;
    if !value.is_finite() {
        return Err(owner.invalid(format!("{}: {name} must be finite", owner.name())));
    }
    Ok(value)
}

fn refline_x_span(
    handle: FigureHandle,
    axes_index: usize,
    owner: ReferenceLineOwner,
) -> BuiltinResult<(f64, f64)> {
    if let Some(limits) = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| owner.internal(format!("{}: {err}", owner.name())))?
        .x_limits
        .filter(|limits| valid_span(*limits))
    {
        return Ok(limits);
    }

    if let Some(mut figure) = clone_figure(handle) {
        let bounds = figure.data_bounds_for_axes(axes_index);
        let span = finite_or_padded_span(bounds.min.x as f64, bounds.max.x as f64);
        if let Some(span) = span {
            return Ok(span);
        }
    }

    Ok((0.0, 1.0))
}

async fn least_squares_specs(
    handle: FigureHandle,
    axes_index: usize,
    owner: ReferenceLineOwner,
) -> BuiltinResult<Vec<ReflineLineSpec>> {
    let x_span = refline_x_span(handle, axes_index, owner)?;
    let Some(figure) = clone_figure(handle) else {
        return Ok(Vec::new());
    };

    let mut specs = Vec::new();
    for (plot_index, plot) in figure.plots().enumerate() {
        if figure
            .plot_axes_indices()
            .get(plot_index)
            .copied()
            .unwrap_or(0)
            != axes_index
        {
            continue;
        }
        if !plot.is_visible() {
            continue;
        }

        let data = match plot {
            PlotElement::Scatter(plot) => {
                Some(plot.export_scene_xy_data().await.map_err(|err| {
                    owner.internal(format!(
                        "{}: unable to read scatter data: {err}",
                        owner.name()
                    ))
                })?)
            }
            PlotElement::Line(plot)
                if plot.marker.is_some() && matches!(plot.line_style, LineStyle::None) =>
            {
                Some(plot.export_scene_xy_data().await.map_err(|err| {
                    owner.internal(format!("{}: unable to read line data: {err}", owner.name()))
                })?)
            }
            _ => None,
        };
        let Some((x, y)) = data else {
            continue;
        };
        if let Some((slope, intercept)) = least_squares_coefficients(&x, &y, owner)? {
            specs.push(ReflineLineSpec {
                slope,
                intercept,
                x_span,
            });
        }
    }

    Ok(specs)
}

fn line_from_spec(
    spec: ReflineLineSpec,
    style: &crate::builtins::plotting::style::ParsedLineStyle,
    owner: ReferenceLineOwner,
) -> BuiltinResult<LinePlot> {
    let y_span = (
        spec.slope * spec.x_span.0 + spec.intercept,
        spec.slope * spec.x_span.1 + spec.intercept,
    );
    if !y_span.0.is_finite() || !y_span.1.is_finite() {
        return Err(owner.invalid(format!(
            "{}: computed line coordinates must be finite",
            owner.name()
        )));
    }

    let mut line = LinePlot::new(vec![spec.x_span.0, spec.x_span.1], vec![y_span.0, y_span.1])
        .map_err(|err| owner.internal(format!("{}: {err}", owner.name())))?
        .with_style(
            style.appearance.color,
            style.appearance.line_width,
            style.appearance.line_style,
        );
    line.set_marker(marker_metadata_from_appearance(&style.appearance));
    if let Some(visible) = style.appearance.visible {
        line.set_visible(visible);
    }
    if let Some(label) = style.label.clone() {
        line = line.with_label(label);
    }
    Ok(line)
}

fn finite_or_padded_span(min: f64, max: f64) -> Option<(f64, f64)> {
    if !min.is_finite() || !max.is_finite() {
        return None;
    }
    if min < max {
        return Some((min, max));
    }
    if min == max {
        let pad = min.abs().max(1.0) * 0.5;
        return Some((min - pad, max + pad));
    }
    None
}

fn valid_span((min, max): (f64, f64)) -> bool {
    min.is_finite() && max.is_finite() && min < max
}

#[cfg(test)]
mod tests;
