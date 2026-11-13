use runmat_value::{CellArray, CharArray, Value};

use super::axis_ticks::TickAxis;
use super::op_common::axes_target::prepare_axes_call;
use super::state::{
    axis_display_bounds_snapshot_for_axes_for_property,
    axis_tick_formats_snapshot_for_axes_for_property,
    axis_tick_labels_snapshot_for_axes_for_property, axis_ticks_snapshot_for_axes_for_property,
    set_axis_tick_labels_for_axes_for_property, set_axis_ticks_for_axes_for_property,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::common::tensor;
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TickLabelMode {
    Auto,
    Manual,
}

pub fn axis_tick_labels_builtin(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    axis: TickAxis,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let (target, args) = prepare_axes_call(runtime, builtin, args.view())?;
    match (args.len(), args.first()) {
        (0, _) => query_labels(builtin, axis, &target.resolve(builtin)?),
        (1, Some(value)) => {
            if let Some(text) = value_as_string(value).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    builtin,
                    source.to_string(),
                    source,
                )
            })? {
                match text.trim().to_ascii_lowercase().as_str() {
                    "auto" => {
                        super::op_common::axes_output::validate_setter_output(builtin)?;
                        apply_mode(
                            builtin,
                            axis,
                            &target.resolve(builtin)?,
                            TickLabelMode::Auto,
                        )?;
                        return Ok(Value::String("auto".into()));
                    }
                    "manual" => {
                        super::op_common::axes_output::validate_setter_output(builtin)?;
                        apply_mode(
                            builtin,
                            axis,
                            &target.resolve(builtin)?,
                            TickLabelMode::Manual,
                        )?;
                        return Ok(Value::String("manual".into()));
                    }
                    "mode" => return query_mode(builtin, axis, &target.resolve(builtin)?),
                    _ => {}
                }
            }
            let labels = labels_from_value(value, builtin)?;
            super::op_common::axes_output::validate_setter_output(builtin)?;
            Ok(label_value(set_labels(
                builtin,
                axis,
                &target.resolve(builtin)?,
                labels,
            )?))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick-label argument"),
        )),
    }
}

fn query_labels(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    Ok(label_value(current_labels(builtin, axis, target)?))
}

fn query_mode(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    let manual = {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        match axis {
            TickAxis::X => x_labels.is_some(),
            TickAxis::Y => y_labels.is_some(),
        }
    };
    Ok(Value::String(if manual { "manual" } else { "auto" }.into()))
}

fn apply_mode(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
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
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Vec<String>> {
    let explicit = {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        match axis {
            TickAxis::X => x_labels,
            TickAxis::Y => y_labels,
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
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Option<String>> {
    let (handle, axes_index) = target.coordinates();
    let formats =
        axis_tick_formats_snapshot_for_axes_for_property(target.authority(), handle, axes_index)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
    Ok(match axis {
        TickAxis::X => formats.0,
        TickAxis::Y => formats.1,
    })
}

fn current_ticks(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Vec<f64>> {
    let (ticks, bounds) = {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let (x_ticks, y_ticks) = axis_ticks_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        let bounds =
            axis_display_bounds_snapshot_for_axes_for_property(property, handle, axes_index)
                .map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
        (
            axis_ticks(axis, x_ticks, y_ticks),
            axis_bounds(bounds, axis),
        )
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
    target: &super::state::ResolvedAxesTarget,
    labels: Vec<String>,
) -> BuiltinResult<Vec<String>> {
    let ticks = current_ticks(builtin, axis, target)?;
    let labels = labels_padded_to_ticks(labels, ticks.len());
    {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let stored_labels = labels.clone();
        let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        let (x_ticks, y_ticks) = axis_ticks_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        let (labels_x, labels_y) = match axis {
            TickAxis::X => (Some(labels), y_labels),
            TickAxis::Y => (x_labels, Some(labels)),
        };
        let (ticks_x, ticks_y) = match axis {
            TickAxis::X => (Some(ticks), y_ticks),
            TickAxis::Y => (x_ticks, Some(ticks)),
        };
        set_axis_ticks_for_axes_for_property(property, handle, axes_index, ticks_x, ticks_y)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        set_axis_tick_labels_for_axes_for_property(
            property, handle, axes_index, labels_x, labels_y,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        Ok(stored_labels)
    }
}

fn clear_labels(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<()> {
    {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let (x_labels, y_labels) = axis_tick_labels_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        let (x, y) = match axis {
            TickAxis::X => (None, y_labels),
            TickAxis::Y => (x_labels, None),
        };
        set_axis_tick_labels_for_axes_for_property(property, handle, axes_index, x, y)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))
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
        Value::String(runmat_value::StringElement::Text(s)) => Ok(vec![s.as_str().to_owned()]),
        Value::StringArray(strings) => strings.data().iter().map(|text| text.as_str().map(str::to_owned).ok_or_else(|| plotting_error(builtin, "tick labels must be text"))).collect(),
        Value::CharArray(chars) => Ok(char_array_rows(chars).map_err(|source|crate::builtins::plotting::plotting_error_with_source(builtin,source.to_string(),source))?),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data().len());
            for entry in cell.data() {
                labels.extend(labels_from_value(entry, builtin)?);
            }
            Ok(labels)
        }
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0 => Ok(Vec::new()),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: tick labels must be a string array or cell array of text, got {other:?}"),
        )),
    }
}

fn char_array_rows(chars: &CharArray) -> Result<Vec<String>, runmat_value::Utf16DecodeError> {
    if chars.rows == 0 || chars.cols == 0 {
        return Ok(Vec::new());
    }
    (0..chars.rows)
        .map(|row| {
            let start = row * chars.cols;
            let end = start + chars.cols;
            Ok(
                runmat_value::decode_utf16_code_units(&chars.data()[start..end])?
                    .trim_end()
                    .to_string(),
            )
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

fn value_as_string(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::CharArray(chars) if chars.rows == 1 => {
            Some(runmat_value::decode_utf16_code_units(chars.data())?)
        }
        _ => None,
    })
}

#[cfg(test)]
pub(crate) fn label_cell_texts(value: &Value) -> Vec<String> {
    let Value::Cell(cell) = value else {
        panic!("expected cell labels, got {value:?}");
    };
    cell.data()
        .iter()
        .map(|entry| match entry {
            Value::CharArray(chars) => runmat_value::decode_utf16_code_units(chars.data())
                .expect("fixture label is well-formed UTF-16"),
            other => panic!("expected char label, got {other:?}"),
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn tensor(data: Vec<f64>) -> Value {
    let len = data.len();
    Value::Tensor(runmat_value::Tensor::new(data, vec![1, len]).expect("tick label test row"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn tick_label_parser_accepts_empty_typed_integer_tensor_without_mirror() {
        let empty =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 0]).expect("empty");

        assert_eq!(
            labels_from_value(&Value::Tensor(empty), "xticklabels").expect("labels"),
            Vec::<String>::new()
        );
    }
}
