use runmat_value::Value;

use super::axis_ticks::TickAxis;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    axis_tick_angles_snapshot_for_axes_for_property, decode_axes_handle,
    set_axis_tick_angles_for_axes_for_property,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::common::tensor;
use crate::BuiltinResult;

enum TickAngleTarget {
    Current,
    Axes(Vec<(super::state::FigureHandle, usize)>),
    Native(super::op_common::axes_target::PreparedAxesTarget),
}

pub fn axis_tick_angle_builtin(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    axis: TickAxis,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let (target, args) = prepare_call(runtime, builtin, args)?;
    match args.len() {
        0 => {
            let mut targets = prepared_targets(runtime, target).into_iter();
            let first = targets.next().expect("current or nonempty admitted axes");
            if targets.next().is_some() {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: query form requires a scalar axes handle"),
                ));
            }
            query_angle(builtin, axis, &first.resolve(builtin)?)
        }
        1 => {
            let value = args.first().expect("one angle argument");
            let angle = angle_from_value(value, builtin)?;
            super::op_common::axes_output::validate_setter_output(builtin)?;
            for target in prepared_targets(runtime, target) {
                set_angle(builtin, axis, &target.resolve(builtin)?, angle)?;
            }
            Ok(Value::Num(angle))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick-angle argument"),
        )),
    }
}

fn prepare_call(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(TickAngleTarget, crate::call::BuiltinArguments)> {
    if let Some(first) = args
        .first()
        .filter(|value| matches!(value, Value::Object(_) | Value::HandleObject(_)))
    {
        if let Some(prepared) =
            super::op_common::axes_target::prepare_graphics_target(runtime, builtin, first)?
        {
            return Ok((
                TickAngleTarget::Native(prepared.into_axes(builtin)?),
                args.into_range(1..).expect("nonempty axes arguments"),
            ));
        }
    }
    split_optional_axes_target(builtin, args)
}

fn prepared_targets(
    runtime: Option<&crate::context::RuntimeContext>,
    target: TickAngleTarget,
) -> Vec<super::op_common::axes_target::PreparedAxesTarget> {
    use super::op_common::axes_target::PreparedAxesTarget;
    match target {
        TickAngleTarget::Current => vec![PreparedAxesTarget::current(runtime)],
        TickAngleTarget::Axes(targets) => targets
            .into_iter()
            .map(|(handle, index)| PreparedAxesTarget::alias(runtime, handle, index))
            .collect(),
        TickAngleTarget::Native(target) => vec![target],
    }
}

fn split_optional_axes_target(
    builtin: &'static str,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(TickAngleTarget, crate::call::BuiltinArguments)> {
    let Some(first) = args.first() else {
        return Ok((TickAngleTarget::Current, args));
    };
    match resolve_plot_handle(first, builtin) {
        Ok(PlotHandle::Axes(handle, axes_index)) => {
            return Ok((
                TickAngleTarget::Axes(vec![(handle, axes_index)]),
                args.into_range(1..).expect("nonempty axes arguments"),
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
    if let Some(targets) = axes_array_targets(first) {
        return Ok((
            TickAngleTarget::Axes(targets),
            args.into_range(1..).expect("nonempty axes arguments"),
        ));
    }
    Ok((TickAngleTarget::Current, args))
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
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    let (handle, index) = target.coordinates();
    let (x, y) = axis_tick_angles_snapshot_for_axes_for_property(target.authority(), handle, index)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
    Ok(Value::Num(match axis {
        TickAxis::X => x.unwrap_or(0.0),
        TickAxis::Y => y.unwrap_or(0.0),
    }))
}

fn set_angle(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
    angle: f64,
) -> BuiltinResult<()> {
    let (handle, index) = target.coordinates();
    let property = target.authority();
    let (x, y) = axis_tick_angles_snapshot_for_axes_for_property(property, handle, index)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
    let (x, y) = match axis {
        TickAxis::X => (Some(angle), y),
        TickAxis::Y => (x, Some(angle)),
    };
    set_axis_tick_angles_for_axes_for_property(property, handle, index, x, y)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))
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
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn axis_tick_angle_scalar_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![45]), vec![1, 1]).unwrap();

        assert_eq!(scalar_numeric_value(&Value::Tensor(tensor)), Some(45.0));
    }
}
