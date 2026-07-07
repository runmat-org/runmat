use runmat_builtins::{CellArray, CharArray, Value};

use super::axis_ticks::TickAxis;
use super::op_common::axes_target::AxesTarget;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    axis_display_bounds_snapshot, axis_display_bounds_snapshot_for_axes,
    axis_tick_formats_snapshot, axis_tick_formats_snapshot_for_axes, axis_tick_labels_snapshot,
    axis_tick_labels_snapshot_for_axes, axis_ticks_snapshot, axis_ticks_snapshot_for_axes,
    set_axis_tick_labels, set_axis_tick_labels_for_axes, set_axis_ticks, set_axis_ticks_for_axes,
};
use super::{plotting_error, plotting_error_with_source};
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TickLabelMode {
    Auto,
    Manual,
}

pub fn axis_tick_labels_builtin(
    builtin: &'static str,
    axis: TickAxis,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(builtin, args)?;
    match args.as_slice() {
        [] => query_labels(builtin, axis, target),
        [value] => {
            if let Some(text) = value_as_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "auto" => {
                        apply_mode(builtin, axis, target, TickLabelMode::Auto)?;
                        return Ok(Value::String("auto".into()));
                    }
                    "manual" => {
                        apply_mode(builtin, axis, target, TickLabelMode::Manual)?;
                        return Ok(Value::String("manual".into()));
                    }
                    "mode" => return query_mode(builtin, axis, target),
                    _ => {}
                }
            }
            let labels = labels_from_value(value, builtin)?;
            Ok(label_value(set_labels(builtin, axis, target, labels)?))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick-label argument"),
        )),
    }
}

fn split_optional_axes_target(
    builtin: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(AxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, builtin) {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

fn query_labels(builtin: &'static str, axis: TickAxis, target: AxesTarget) -> BuiltinResult<Value> {
    Ok(label_value(current_labels(builtin, axis, target)?))
}

fn query_mode(builtin: &'static str, axis: TickAxis, target: AxesTarget) -> BuiltinResult<Value> {
    let manual = match target {
        Some((handle, axes_index)) => {
            let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes(handle, axes_index)
                .map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            match axis {
                TickAxis::X => x_labels.is_some(),
                TickAxis::Y => y_labels.is_some(),
            }
        }
        None => {
            let (x_labels, y_labels) = axis_tick_labels_snapshot();
            match axis {
                TickAxis::X => x_labels.is_some(),
                TickAxis::Y => y_labels.is_some(),
            }
        }
    };
    Ok(Value::String(if manual { "manual" } else { "auto" }.into()))
}

fn apply_mode(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesTarget,
    mode: TickLabelMode,
) -> BuiltinResult<()> {
    match mode {
        TickLabelMode::Auto => clear_labels(builtin, axis, target),
        TickLabelMode::Manual => {
            let labels = current_labels(builtin, axis, target)?;
            set_labels(builtin, axis, target, labels)?;
            Ok(())
        }
    }
}

fn current_labels(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesTarget,
) -> BuiltinResult<Vec<String>> {
    let explicit = match target {
        Some((handle, axes_index)) => {
            let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes(handle, axes_index)
                .map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            match axis {
                TickAxis::X => x_labels,
                TickAxis::Y => y_labels,
            }
        }
        None => {
            let (x_labels, y_labels) = axis_tick_labels_snapshot();
            match axis {
                TickAxis::X => x_labels,
                TickAxis::Y => y_labels,
            }
        }
    };
    if let Some(labels) = explicit {
        return Ok(labels);
    }
    let format = current_format(builtin, axis, target)?;
    let formatter = runmat_plot::core::plot_utils::TickLabelFormatter::new(format.as_deref());
    Ok(current_ticks(builtin, axis, target)?
        .into_iter()
        .map(|tick| formatter.format(tick))
        .collect())
}

fn current_format(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesTarget,
) -> BuiltinResult<Option<String>> {
    let formats = match target {
        Some((handle, axes_index)) => axis_tick_formats_snapshot_for_axes(handle, axes_index)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?,
        None => axis_tick_formats_snapshot(),
    };
    Ok(match axis {
        TickAxis::X => formats.0,
        TickAxis::Y => formats.1,
    })
}

fn current_ticks(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesTarget,
) -> BuiltinResult<Vec<f64>> {
    let (ticks, bounds) = match target {
        Some((handle, axes_index)) => {
            let (x_ticks, y_ticks) =
                axis_ticks_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            let bounds =
                axis_display_bounds_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            (
                axis_ticks(axis, x_ticks, y_ticks),
                axis_bounds(bounds, axis),
            )
        }
        None => {
            let (x_ticks, y_ticks) = axis_ticks_snapshot();
            let bounds = axis_display_bounds_snapshot();
            (
                axis_ticks(axis, x_ticks, y_ticks),
                axis_bounds(bounds, axis),
            )
        }
    };
    Ok(ticks.unwrap_or_else(|| {
        let (lo, hi) = bounds.unwrap_or((-1.0, 1.0));
        runmat_plot::core::plot_utils::generate_major_ticks(lo, hi)
    }))
}

fn axis_ticks(
    axis: TickAxis,
    x_ticks: Option<Vec<f64>>,
    y_ticks: Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    match axis {
        TickAxis::X => x_ticks,
        TickAxis::Y => y_ticks,
    }
}

fn axis_bounds(bounds: Option<(f64, f64, f64, f64)>, axis: TickAxis) -> Option<(f64, f64)> {
    bounds.map(|(x_min, x_max, y_min, y_max)| match axis {
        TickAxis::X => (x_min, x_max),
        TickAxis::Y => (y_min, y_max),
    })
}

fn set_labels(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesTarget,
    labels: Vec<String>,
) -> BuiltinResult<Vec<String>> {
    let ticks = current_ticks(builtin, axis, target)?;
    let labels = labels_padded_to_ticks(labels, ticks.len());
    match target {
        Some((handle, axes_index)) => {
            let stored_labels = labels.clone();
            let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes(handle, axes_index)
                .map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            let (x_ticks, y_ticks) =
                axis_ticks_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            let (labels_x, labels_y) = match axis {
                TickAxis::X => (Some(labels), y_labels),
                TickAxis::Y => (x_labels, Some(labels)),
            };
            let (ticks_x, ticks_y) = match axis {
                TickAxis::X => (Some(ticks), y_ticks),
                TickAxis::Y => (x_ticks, Some(ticks)),
            };
            set_axis_ticks_for_axes(handle, axes_index, ticks_x, ticks_y).map_err(|err| {
                plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
            })?;
            set_axis_tick_labels_for_axes(handle, axes_index, labels_x, labels_y).map_err(
                |err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err),
            )?;
            Ok(stored_labels)
        }
        None => {
            let stored_labels = labels.clone();
            let (x_labels, y_labels) = axis_tick_labels_snapshot();
            let (x_ticks, y_ticks) = axis_ticks_snapshot();
            let (labels_x, labels_y) = match axis {
                TickAxis::X => (Some(labels), y_labels),
                TickAxis::Y => (x_labels, Some(labels)),
            };
            let (ticks_x, ticks_y) = match axis {
                TickAxis::X => (Some(ticks), y_ticks),
                TickAxis::Y => (x_ticks, Some(ticks)),
            };
            set_axis_ticks(ticks_x, ticks_y);
            set_axis_tick_labels(labels_x, labels_y);
            Ok(stored_labels)
        }
    }
}

fn clear_labels(builtin: &'static str, axis: TickAxis, target: AxesTarget) -> BuiltinResult<()> {
    match target {
        Some((handle, axes_index)) => {
            let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes(handle, axes_index)
                .map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            let (x, y) = match axis {
                TickAxis::X => (None, y_labels),
                TickAxis::Y => (x_labels, None),
            };
            set_axis_tick_labels_for_axes(handle, axes_index, x, y).map_err(|err| {
                plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
            })
        }
        None => {
            let (x_labels, y_labels) = axis_tick_labels_snapshot();
            let (x, y) = match axis {
                TickAxis::X => (None, y_labels),
                TickAxis::Y => (x_labels, None),
            };
            set_axis_tick_labels(x, y);
            Ok(())
        }
    }
}

fn labels_padded_to_ticks(mut labels: Vec<String>, tick_count: usize) -> Vec<String> {
    if labels.len() < tick_count {
        labels.resize(tick_count, String::new());
    }
    labels
}

fn labels_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::StringArray(strings) => Ok(strings.data.clone()),
        Value::CharArray(chars) => Ok(char_array_rows(chars)),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for entry in &cell.data {
                labels.extend(labels_from_value(entry, builtin)?);
            }
            Ok(labels)
        }
        Value::Tensor(tensor) if tensor.data.is_empty() => Ok(Vec::new()),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: tick labels must be a string array or cell array of text, got {other:?}"),
        )),
    }
}

fn char_array_rows(chars: &CharArray) -> Vec<String> {
    if chars.rows == 0 || chars.cols == 0 {
        return Vec::new();
    }
    (0..chars.rows)
        .map(|row| {
            let start = row * chars.cols;
            let end = start + chars.cols;
            chars.data[start..end]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

fn label_value(labels: Vec<String>) -> Value {
    let values = labels
        .iter()
        .map(|label| Value::CharArray(CharArray::new_row(label)))
        .collect::<Vec<_>>();
    Value::Cell(CellArray::new(values, 1, labels.len()).expect("valid tick-label cell array"))
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn label_cell_texts(value: &Value) -> Vec<String> {
    let Value::Cell(cell) = value else {
        panic!("expected cell labels, got {value:?}");
    };
    cell.data
        .iter()
        .map(|entry| match entry {
            Value::CharArray(chars) => chars.data.iter().collect(),
            other => panic!("expected char label, got {other:?}"),
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn tensor(data: Vec<f64>) -> Value {
    Value::Tensor(runmat_builtins::Tensor {
        rows: 1,
        cols: data.len(),
        shape: vec![1, data.len()],
        data,
        dtype: runmat_builtins::NumericDType::F64,
    })
}
