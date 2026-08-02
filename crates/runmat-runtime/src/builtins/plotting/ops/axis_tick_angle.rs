use runmat_builtins::Value;

use super::axis_ticks::TickAxis;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    axis_tick_angles_snapshot, axis_tick_angles_snapshot_for_axes, decode_axes_handle,
    set_axis_tick_angles, set_axis_tick_angles_for_axes,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::common::tensor;
use crate::BuiltinResult;

#[derive(Clone, Debug)]
enum TickAngleTarget {
    Current,
    Axes(Vec<(super::state::FigureHandle, usize)>),
}

pub fn axis_tick_angle_builtin(
    builtin: &'static str,
    axis: TickAxis,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(builtin, args)?;
    match args.as_slice() {
        [] => query_angle(builtin, axis, target),
        [value] => {
            let angle = angle_from_value(value, builtin)?;
            set_angle(builtin, axis, target, angle)?;
            Ok(Value::Num(angle))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick-angle argument"),
        )),
    }
}

fn split_optional_axes_target(
    builtin: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(TickAngleTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((TickAngleTarget::Current, Vec::new()));
    };
    match resolve_plot_handle(&first, builtin) {
        Ok(PlotHandle::Axes(handle, axes_index)) => {
            return Ok((
                TickAngleTarget::Axes(vec![(handle, axes_index)]),
                iter.collect(),
            ));
        }
        Ok(
            PlotHandle::Ruler(..)
            | PlotHandle::Text(..)
            | PlotHandle::Legend(..)
            | PlotHandle::PlotChild(_, _),
        ) => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: expected axes handle or tick angle"),
            ));
        }
        Ok(PlotHandle::Root | PlotHandle::Figure(_)) | Err(_) => {}
    }
    if let Some(targets) = axes_array_targets(&first) {
        return Ok((TickAngleTarget::Axes(targets), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((TickAngleTarget::Current, rest))
}

fn axes_array_targets(value: &Value) -> Option<Vec<(super::state::FigureHandle, usize)>> {
    let Value::Tensor(tensor) = value else {
        return None;
    };
    let data = tensor::tensor_values_f64(tensor);
    if data.is_empty() {
        return None;
    }
    let mut targets = Vec::with_capacity(data.len());
    for scalar in &data {
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

fn query_angle(
    builtin: &'static str,
    axis: TickAxis,
    target: TickAngleTarget,
) -> BuiltinResult<Value> {
    let (x_angle, y_angle) = match target {
        TickAngleTarget::Axes(targets) if targets.len() == 1 => {
            let (handle, axes_index) = targets[0];
            axis_tick_angles_snapshot_for_axes(handle, axes_index).map_err(|err| {
                plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
            })?
        }
        TickAngleTarget::Axes(_) => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: query form requires a scalar axes handle"),
            ))
        }
        TickAngleTarget::Current => axis_tick_angles_snapshot(),
    };
    Ok(Value::Num(match axis {
        TickAxis::X => x_angle.unwrap_or(0.0),
        TickAxis::Y => y_angle.unwrap_or(0.0),
    }))
}

fn set_angle(
    builtin: &'static str,
    axis: TickAxis,
    target: TickAngleTarget,
    angle: f64,
) -> BuiltinResult<()> {
    match target {
        TickAngleTarget::Axes(targets) => {
            for (handle, axes_index) in targets {
                let (x_angle, y_angle) = axis_tick_angles_snapshot_for_axes(handle, axes_index)
                    .map_err(|err| {
                        plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                    })?;
                let (x, y) = match axis {
                    TickAxis::X => (Some(angle), y_angle),
                    TickAxis::Y => (x_angle, Some(angle)),
                };
                set_axis_tick_angles_for_axes(handle, axes_index, x, y).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            }
            Ok(())
        }
        TickAngleTarget::Current => {
            let (x_angle, y_angle) = axis_tick_angles_snapshot();
            let (x, y) = match axis {
                TickAxis::X => (Some(angle), y_angle),
                TickAxis::Y => (x_angle, Some(angle)),
            };
            set_axis_tick_angles(x, y);
            Ok(())
        }
    }
}

fn angle_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<f64> {
    let angle = scalar_numeric_value(value)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: tick angle must be numeric")))?;
    if angle.is_finite() {
        Ok(angle)
    } else {
        Err(plotting_error(
            builtin,
            format!("{builtin}: tick angle must be finite"),
        ))
    }
}

fn scalar_numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntegerStorage, Tensor};

    #[test]
    fn axis_tick_angle_scalar_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![45]), vec![1, 1]).unwrap();

        assert_eq!(scalar_numeric_value(&Value::Tensor(tensor)), Some(45.0));
    }
}
