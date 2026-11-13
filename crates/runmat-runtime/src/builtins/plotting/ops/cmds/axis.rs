use super::error::cmd_error_with_message;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    axes_metadata_snapshot, axis_color_limits_snapshot_for_axes,
    axis_display_bounds_snapshot_for_axes, axis_z_limits_snapshot_for_axes, current_axes_state,
    current_axes_target_before_mutation, current_figure_handle_if_exists, set_axes_style_for_axes,
    set_axis_equal, set_axis_equal_and_limits, set_axis_limits, set_color_limits_runtime,
    set_z_limits, FigureHandle,
};
use crate::RuntimeError;
use runmat_builtins::{
    AXIS_DESCRIPTOR, AXIS_ERROR_INVALID_ARGUMENT, AXIS_ERROR_OUTPUTS, AXIS_STATE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
#[runtime_builtin(
    name = "axis",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds::axis"
)]
pub fn axis_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(&AXIS_DESCRIPTOR, &AXIS_ERROR_OUTPUTS, "axis")?;

    if args.is_empty() {
        return axis_query_value(None);
    }
    if args.len() != 1 {
        return Err(axis_invalid_argument());
    }
    if crate::output_count::current_output_count() == Some(1)
        && !matches!(
            resolve_plot_handle(&args[0], "axis"),
            Ok(PlotHandle::Axes(_, _))
        )
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &AXIS_STATE_OUTPUT_EXTENSION,
            "axis",
        )?;
    }

    if let Value::Tensor(t) = &args[0] {
        if matches!(tensor_utils::tensor_element_len(t), 4 | 6 | 8) {
            let values = tensor_utils::tensor_values_f64(t);
            let state = current_figure_handle_if_exists().map(|_| current_axes_state());
            let bounds = state
                .as_ref()
                .and_then(|state| {
                    axis_display_bounds_snapshot_for_axes(state.handle, state.active_index)
                        .ok()
                        .flatten()
                })
                .unwrap_or((0.0, 1.0, 0.0, 1.0));
            let x = axis_limit_pair(values[0], values[1], bounds.0, bounds.1)?;
            let y = axis_limit_pair(values[2], values[3], bounds.2, bounds.3)?;
            let z = if values.len() >= 6 {
                let z_auto = state
                    .as_ref()
                    .map(|state| axis_z_limits_snapshot_for_axes(state.handle, state.active_index))
                    .transpose()
                    .map_err(|_| axis_invalid_argument())?
                    .flatten()
                    .unwrap_or((0.0, 1.0));
                Some(axis_limit_pair(values[4], values[5], z_auto.0, z_auto.1)?)
            } else {
                None
            };
            let c = if values.len() == 8 {
                let c_auto = if values[6] == f64::NEG_INFINITY || values[7] == f64::INFINITY {
                    let range = state
                        .as_ref()
                        .map(|state| {
                            axis_color_limits_snapshot_for_axes(state.handle, state.active_index)
                        })
                        .transpose()
                        .map_err(|_| axis_invalid_argument())?
                        .unwrap_or(runmat_plot::plots::surface::ScalarColorRange::Empty);
                    match range {
                        runmat_plot::plots::surface::ScalarColorRange::Known(lower, upper) => (lower, upper),
                        runmat_plot::plots::surface::ScalarColorRange::Empty | runmat_plot::plots::surface::ScalarColorRange::NotMapped => (0., 1.),
                        runmat_plot::plots::surface::ScalarColorRange::Unavailable => return Err(cmd_error_with_message("axis", "axis: automatic color limits are unavailable for this resident plot", &AXIS_ERROR_INVALID_ARGUMENT)),
                    }
                } else {
                    (0., 1.)
                };
                Some(axis_limit_pair(values[6], values[7], c_auto.0, c_auto.1)?)
            } else {
                None
            };
            // Admit every requested pair before publishing any limit change.
            set_axis_limits(Some(x), Some(y));
            if let Some(z) = z {
                set_z_limits(Some(z));
            }
            if let Some(c) = c {
                set_color_limits_runtime(Some(c));
            }
            return Ok(Value::Bool(true));
        }
    }

    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&args[0], "axis") {
        return axis_query_value(Some((handle, axes_index)));
    }

    if let Some(visible) = axis_visibility_value(&args[0])? {
        set_current_axis_visibility(visible)?;
        return Ok(Value::Bool(true));
    }

    let Some(mode) = as_lower_str(&args[0]).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source("cmds", source.to_string(), source)
    })?
    else {
        return Err(axis_invalid_argument());
    };
    match mode.trim() {
        "equal" => {
            set_axis_equal(true);
            Ok(Value::Bool(true))
        }
        "auto" => {
            set_axis_equal_and_limits(false, None, None);
            set_z_limits(None);
            Ok(Value::Bool(true))
        }
        "tight" => {
            // Treat as auto; camera fit uses data bounds.
            set_axis_limits(None, None);
            set_z_limits(None);
            Ok(Value::Bool(true))
        }
        "image" => {
            set_axis_equal_and_limits(true, None, None);
            Ok(Value::Bool(true))
        }
        "on" => {
            set_current_axis_visibility(true)?;
            Ok(Value::Bool(true))
        }
        "off" => {
            set_current_axis_visibility(false)?;
            Ok(Value::Bool(true))
        }
        "manual" => {
            let (handle, axes_index) = current_axes_target_before_mutation();
            crate::builtins::plotting::state::lock_axes_limits(handle, axes_index)
                .map_err(|_| axis_invalid_argument())?;
            Ok(Value::Bool(true))
        }
        "ij" | "xy" => {
            let state = current_axes_state();
            let direction = if mode.trim() == "ij" {
                runmat_plot::plots::figure::YDirection::Reverse
            } else {
                runmat_plot::plots::figure::YDirection::Normal
            };
            crate::builtins::plotting::state::set_y_direction_for_axes(
                state.handle,
                state.active_index,
                direction,
            )
            .map_err(|_| axis_invalid_argument())?;
            Ok(Value::Bool(true))
        }
        other => Err(cmd_error_with_message(
            "axis",
            format!(
                "{}: unsupported argument '{other}'",
                AXIS_ERROR_INVALID_ARGUMENT.message
            ),
            &AXIS_ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn axis_invalid_argument() -> RuntimeError {
    cmd_error_with_message(
        "axis",
        AXIS_ERROR_INVALID_ARGUMENT.message,
        &AXIS_ERROR_INVALID_ARGUMENT,
    )
}

fn axis_limit_pair(
    lower: f64,
    upper: f64,
    automatic_lower: f64,
    automatic_upper: f64,
) -> crate::BuiltinResult<(f64, f64)> {
    let lower = if lower == f64::NEG_INFINITY {
        automatic_lower
    } else if lower.is_finite() {
        lower
    } else {
        return Err(axis_invalid_argument());
    };
    let upper = if upper == f64::INFINITY {
        automatic_upper
    } else if upper.is_finite() {
        upper
    } else {
        return Err(axis_invalid_argument());
    };
    if upper <= lower {
        return Err(axis_invalid_argument());
    }
    Ok((lower, upper))
}

fn axis_visibility_value(value: &Value) -> crate::BuiltinResult<Option<bool>> {
    let scalar = match value {
        Value::Bool(value) => return Ok(Some(*value)),
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Tensor(tensor) if tensor_utils::tensor_element_len(tensor) == 1 => {
            Some(tensor_utils::tensor_values_f64(tensor)[0])
        }
        _ => None,
    };
    match scalar {
        Some(0.0) => Ok(Some(false)),
        Some(1.0) => Ok(Some(true)),
        Some(_) => Err(axis_invalid_argument()),
        None => Ok(None),
    }
}

fn set_current_axis_visibility(visible: bool) -> crate::BuiltinResult<()> {
    let state = current_axes_state();
    let mut style = axes_metadata_snapshot(state.handle, state.active_index)
        .map_err(|_| axis_invalid_argument())?
        .axes_style;
    style.visible = visible;
    set_axes_style_for_axes(state.handle, state.active_index, style)
        .map_err(|_| axis_invalid_argument())
}

fn axis_query_value(target: Option<(FigureHandle, usize)>) -> crate::BuiltinResult<Value> {
    let (handle, axes_index) = target.unwrap_or_else(|| {
        let state = current_axes_state();
        (state.handle, state.active_index)
    });
    let meta = axes_metadata_snapshot(handle, axes_index).map_err(|_| axis_invalid_argument())?;
    let bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
        .map_err(|_| axis_invalid_argument())?
        .unwrap_or((0.0, 1.0, 0.0, 1.0));
    let x = meta.x_limits.unwrap_or((bounds.0, bounds.1));
    let y = meta.y_limits.unwrap_or((bounds.2, bounds.3));
    let mut values = vec![x.0, x.1, y.0, y.1];
    if let Some(z) =
        axis_z_limits_snapshot_for_axes(handle, axes_index).map_err(|_| axis_invalid_argument())?
    {
        values.extend_from_slice(&[z.0, z.1]);
    }
    let len = values.len();
    Ok(Value::Tensor(
        Tensor::new(values, vec![1, len]).expect("axis query row"),
    ))
}
#[cfg(test)]
#[path = "axis/tests.rs"]
mod tests;
