use runmat_builtins::Value;

use super::axis_ticks::TickAxis;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    axis_tick_formats_snapshot, axis_tick_formats_snapshot_for_axes, decode_axes_handle,
    set_axis_tick_formats, set_axis_tick_formats_for_axes,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::plotting::style::value_as_string;
use crate::BuiltinResult;

const DEFAULT_TICK_FORMAT: &str = "%g";

#[derive(Clone, Debug)]
enum TickFormatTarget {
    Current,
    Axes(Vec<(super::state::FigureHandle, usize)>),
}

pub fn axis_tick_format_builtin(
    builtin: &'static str,
    axis: TickAxis,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(builtin, args)?;
    match args.as_slice() {
        [] => query_format(builtin, axis, target),
        [value] => {
            let format = format_from_value(value, builtin)?;
            set_format(builtin, axis, target, format.clone())?;
            Ok(Value::String(format))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick-format argument"),
        )),
    }
}

fn split_optional_axes_target(
    builtin: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(TickFormatTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((TickFormatTarget::Current, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, builtin) {
        return Ok((
            TickFormatTarget::Axes(vec![(handle, axes_index)]),
            iter.collect(),
        ));
    }
    if let Some(targets) = axes_array_targets(&first) {
        return Ok((TickFormatTarget::Axes(targets), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((TickFormatTarget::Current, rest))
}

fn axes_array_targets(value: &Value) -> Option<Vec<(super::state::FigureHandle, usize)>> {
    let Value::Tensor(tensor) = value else {
        return None;
    };
    if tensor.data.is_empty() {
        return None;
    }
    let mut targets = Vec::with_capacity(tensor.data.len());
    for scalar in &tensor.data {
        let Ok((handle, axes_index)) = decode_axes_handle(*scalar) else {
            return None;
        };
        if !super::state::axes_handle_exists(handle, axes_index) {
            return None;
        }
        targets.push((handle, axes_index));
    }
    Some(targets)
}

fn query_format(
    builtin: &'static str,
    axis: TickAxis,
    target: TickFormatTarget,
) -> BuiltinResult<Value> {
    let (x_format, y_format) = match target {
        TickFormatTarget::Axes(targets) if targets.len() == 1 => {
            let (handle, axes_index) = targets[0];
            axis_tick_formats_snapshot_for_axes(handle, axes_index).map_err(|err| {
                plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
            })?
        }
        TickFormatTarget::Axes(_) => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: query form requires a scalar axes handle"),
            ))
        }
        TickFormatTarget::Current => axis_tick_formats_snapshot(),
    };
    let format = match axis {
        TickAxis::X => x_format,
        TickAxis::Y => y_format,
    };
    Ok(Value::String(
        format.unwrap_or_else(|| DEFAULT_TICK_FORMAT.into()),
    ))
}

fn set_format(
    builtin: &'static str,
    axis: TickAxis,
    target: TickFormatTarget,
    format: String,
) -> BuiltinResult<()> {
    match target {
        TickFormatTarget::Axes(targets) => {
            for (handle, axes_index) in targets {
                let (x_format, y_format) = axis_tick_formats_snapshot_for_axes(handle, axes_index)
                    .map_err(|err| {
                        plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                    })?;
                let (x, y) = match axis {
                    TickAxis::X => (Some(format.clone()), y_format),
                    TickAxis::Y => (x_format, Some(format.clone())),
                };
                set_axis_tick_formats_for_axes(handle, axes_index, x, y).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            }
            Ok(())
        }
        TickFormatTarget::Current => {
            let (x_format, y_format) = axis_tick_formats_snapshot();
            let (x, y) = match axis {
                TickAxis::X => (Some(format), y_format),
                TickAxis::Y => (x_format, Some(format)),
            };
            set_axis_tick_formats(x, y);
            Ok(())
        }
    }
}

fn format_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    let format = value_as_string(value).ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: tick format must be a string scalar"),
        )
    })?;
    Ok(runmat_plot::core::plot_renderer::plot_utils::canonical_tick_label_format(&format))
}
