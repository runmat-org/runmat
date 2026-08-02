use runmat_builtins::Value;

use super::axis_ticks::TickAxis;
use super::state::{
    axes_metadata_snapshot, current_axes_state, decode_axes_handle, figure_handle_exists,
    set_log_modes_for_axes, FigureHandle,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::handles::numeric_handle_scalar;
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
    let integer_scalar = tensor_utils::scalar_integer_value(&first);
    if let Some(integer) = integer_scalar {
        // Axes handles occupy at most 52 bits (u32 figure id plus 20-bit axes
        // index), so this bounded conversion to f64 remains exact.
        let encoded = integer.try_to_u64().ok_or_else(|| {
            plotting_error(
                builtin,
                format!("{builtin}: expected axes handle followed by optional scale"),
            )
        })?;
        const MAX_AXES_HANDLE: u64 = ((u32::MAX as u64) << 20) | ((1 << 20) - 1);
        if encoded == 0 || encoded > MAX_AXES_HANDLE {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: expected axes handle followed by optional scale"),
            ));
        }
        let scalar = encoded as f64;
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
    numeric_handle_scalar(value)
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntegerStorage, Tensor};

    fn poisoned_scalar(storage: IntegerStorage) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        Value::Tensor(tensor)
    }

    #[test]
    fn axis_scale_numeric_scalar_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![12]), vec![1, 1]).unwrap();

        assert_eq!(numeric_scalar(&Value::Tensor(tensor)), Some(12.0));
    }

    #[test]
    fn axis_scale_handle_parser_reads_all_integer_storages_without_mirrors() {
        let storages = [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in storages {
            let err = split_optional_axes_target("xscale", vec![poisoned_scalar(storage)])
                .expect_err("unregistered axes handle must be rejected");
            assert!(err.message().contains("expected axes handle"));
        }
    }
}
