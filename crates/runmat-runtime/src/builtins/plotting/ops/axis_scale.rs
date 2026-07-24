use runmat_builtins::Value;

use super::axis_ticks::TickAxis;
use super::state::{
    axes_metadata_snapshot, current_axes_state, decode_axes_handle, figure_handle_exists,
    set_log_modes_for_axes, FigureHandle,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::plotting::style::value_as_string;
use crate::BuiltinResult;

type AxesScaleTarget = Option<(FigureHandle, usize)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisScaleMode {
    Linear,
    Log,
}

impl AxisScaleMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Log => "log",
        }
    }

    pub fn is_log(self) -> bool {
        matches!(self, Self::Log)
    }
}

pub fn axis_scale_builtin(
    builtin: &'static str,
    axis: TickAxis,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(builtin, args)?;
    match args.as_slice() {
        [] => query_scale(builtin, axis, target),
        [value] => {
            let mode = scale_mode_from_value(value, builtin)?;
            set_scale(builtin, axis, target, mode)?;
            Ok(Value::String(mode.as_str().into()))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one scale argument"),
        )),
    }
}

pub fn scale_mode_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<AxisScaleMode> {
    let scale = value_as_string(value)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: scale must be a string")))?;
    match scale.trim().to_ascii_lowercase().as_str() {
        "linear" => Ok(AxisScaleMode::Linear),
        "log" => Ok(AxisScaleMode::Log),
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: scale must be 'linear' or 'log'"),
        )),
    }
}

fn split_optional_axes_target(
    builtin: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(AxesScaleTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Some(scalar) = numeric_scalar(&first) {
        let (handle, axes_index) = decode_axes_handle(scalar).map_err(|_| {
            plotting_error(
                builtin,
                format!("{builtin}: expected axes handle followed by optional scale"),
            )
        })?;
        if !figure_handle_exists(handle) {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: invalid axes handle"),
            ));
        }
        axes_metadata_snapshot(handle, axes_index)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

fn numeric_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(v) => Some(*v),
        Value::Int(i) => Some(i.to_f64()),
        Value::Tensor(t) if t.data.len() == 1 => Some(t.data[0]),
        _ => None,
    }
}

fn query_scale(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesScaleTarget,
) -> BuiltinResult<Value> {
    let (handle, axes_index) = target.unwrap_or_else(|| {
        let current = current_axes_state();
        (current.handle, current.active_index)
    });
    let meta = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
    let is_log = match axis {
        TickAxis::X => meta.x_log,
        TickAxis::Y => meta.y_log,
    };
    Ok(Value::String(if is_log { "log" } else { "linear" }.into()))
}

fn set_scale(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesScaleTarget,
    mode: AxisScaleMode,
) -> BuiltinResult<()> {
    let (handle, axes_index) = target.unwrap_or_else(|| {
        let current = current_axes_state();
        (current.handle, current.active_index)
    });
    let meta = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
    let (x_log, y_log) = match axis {
        TickAxis::X => (mode.is_log(), meta.y_log),
        TickAxis::Y => (meta.x_log, mode.is_log()),
    };
    set_log_modes_for_axes(handle, axes_index, x_log, y_log)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))
}
