use runmat_value::NumericScalar;
use runmat_value::{Tensor, Value};

use super::op_common::axes_target::AxesTarget;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    axis_display_bounds_snapshot, axis_display_bounds_snapshot_for_axes, axis_ticks_snapshot,
    axis_ticks_snapshot_for_axes, set_axis_ticks, set_axis_ticks_for_axes,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::common::tensor;
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickAxis {
    X,
    Y,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TickMode {
    Auto,
    Manual,
}

pub fn axis_ticks_builtin(
    builtin: &'static str,
    axis: TickAxis,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(builtin, args)?;
    match args.as_slice() {
        [] => query_ticks(builtin, axis, target),
        [value] => {
            if let Some(text) = value_as_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "auto" => {
                        apply_mode(builtin, axis, target, TickMode::Auto)?;
                        return Ok(Value::String("auto".into()));
                    }
                    "manual" => {
                        apply_mode(builtin, axis, target, TickMode::Manual)?;
                        return Ok(Value::String("manual".into()));
                    }
                    "mode" => return query_mode(builtin, axis, target),
                    _ => {}
                }
            }
            let ticks = ticks_from_value(value, builtin)?;
            set_ticks(builtin, axis, target, ticks.clone())?;
            Ok(tick_value(ticks))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick argument"),
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

fn query_ticks(builtin: &'static str, axis: TickAxis, target: AxesTarget) -> BuiltinResult<Value> {
    let ticks = match target {
        Some((handle, axes_index)) => {
            let (x_ticks, y_ticks) =
                axis_ticks_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            let bounds =
                axis_display_bounds_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            match axis {
                TickAxis::X => ticks_or_auto(x_ticks.as_deref(), axis_bounds(bounds, axis)),
                TickAxis::Y => ticks_or_auto(y_ticks.as_deref(), axis_bounds(bounds, axis)),
            }
        }
        None => {
            let (x_ticks, y_ticks) = axis_ticks_snapshot();
            let bounds = axis_display_bounds_snapshot();
            match axis {
                TickAxis::X => ticks_or_auto(x_ticks.as_deref(), axis_bounds(bounds, axis)),
                TickAxis::Y => ticks_or_auto(y_ticks.as_deref(), axis_bounds(bounds, axis)),
            }
        }
    };
    Ok(tick_value(ticks))
}

fn query_mode(builtin: &'static str, axis: TickAxis, target: AxesTarget) -> BuiltinResult<Value> {
    let manual = match target {
        Some((handle, axes_index)) => {
            let (x_ticks, y_ticks) =
                axis_ticks_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            match axis {
                TickAxis::X => x_ticks.is_some(),
                TickAxis::Y => y_ticks.is_some(),
            }
        }
        None => {
            let (x_ticks, y_ticks) = axis_ticks_snapshot();
            match axis {
                TickAxis::X => x_ticks.is_some(),
                TickAxis::Y => y_ticks.is_some(),
            }
        }
    };
    Ok(Value::String(if manual { "manual" } else { "auto" }.into()))
}

fn apply_mode(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesTarget,
    mode: TickMode,
) -> BuiltinResult<()> {
    let current = match mode {
        TickMode::Auto => None,
        TickMode::Manual => match target {
            Some((handle, axes_index)) => {
                let (x_ticks, y_ticks) =
                    axis_ticks_snapshot_for_axes(handle, axes_index).map_err(|err| {
                        plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                    })?;
                let bounds =
                    axis_display_bounds_snapshot_for_axes(handle, axes_index).map_err(|err| {
                        plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                    })?;
                Some(match axis {
                    TickAxis::X => ticks_or_auto(x_ticks.as_deref(), axis_bounds(bounds, axis)),
                    TickAxis::Y => ticks_or_auto(y_ticks.as_deref(), axis_bounds(bounds, axis)),
                })
            }
            None => Some({
                let bounds = axis_display_bounds_snapshot();
                match axis {
                    TickAxis::X => {
                        let (x_ticks, _) = axis_ticks_snapshot();
                        ticks_or_auto(x_ticks.as_deref(), axis_bounds(bounds, axis))
                    }
                    TickAxis::Y => {
                        let (_, y_ticks) = axis_ticks_snapshot();
                        ticks_or_auto(y_ticks.as_deref(), axis_bounds(bounds, axis))
                    }
                }
            }),
        },
    };
    match current {
        Some(ticks) => set_ticks(builtin, axis, target, ticks),
        None => clear_ticks(builtin, axis, target),
    }
}

fn set_ticks(
    builtin: &'static str,
    axis: TickAxis,
    target: AxesTarget,
    ticks: Vec<f64>,
) -> BuiltinResult<()> {
    match target {
        Some((handle, axes_index)) => {
            let (x_ticks, y_ticks) =
                axis_ticks_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            let (x, y) = match axis {
                TickAxis::X => (Some(ticks), y_ticks),
                TickAxis::Y => (x_ticks, Some(ticks)),
            };
            set_axis_ticks_for_axes(handle, axes_index, x, y).map_err(|err| {
                plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
            })
        }
        None => {
            let (x_ticks, y_ticks) = axis_ticks_snapshot();
            let (x, y) = match axis {
                TickAxis::X => (Some(ticks), y_ticks),
                TickAxis::Y => (x_ticks, Some(ticks)),
            };
            set_axis_ticks(x, y);
            Ok(())
        }
    }
}

fn clear_ticks(builtin: &'static str, axis: TickAxis, target: AxesTarget) -> BuiltinResult<()> {
    match target {
        Some((handle, axes_index)) => {
            let (x_ticks, y_ticks) =
                axis_ticks_snapshot_for_axes(handle, axes_index).map_err(|err| {
                    plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
                })?;
            let (x, y) = match axis {
                TickAxis::X => (None, y_ticks),
                TickAxis::Y => (x_ticks, None),
            };
            set_axis_ticks_for_axes(handle, axes_index, x, y).map_err(|err| {
                plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
            })
        }
        None => {
            let (x_ticks, y_ticks) = axis_ticks_snapshot();
            let (x, y) = match axis {
                TickAxis::X => (None, y_ticks),
                TickAxis::Y => (x_ticks, None),
            };
            set_axis_ticks(x, y);
            Ok(())
        }
    }
}

fn ticks_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    let tensor =
        Tensor::try_from(value).map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
    let exact = (0..tensor.len())
        .map(|index| {
            tensor
                .numeric_value_at(index)
                .expect("validated tick storage")
        })
        .collect::<Vec<_>>();
    if exact
        .windows(2)
        .any(|pair| !numeric_scalar_strictly_less(pair[0], pair[1]))
    {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: tick values must be strictly increasing"),
        ));
    }
    let ticks = tensor::tensor_values_f64(&tensor);
    if ticks.iter().any(|value| !value.is_finite()) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: tick values must be finite"),
        ));
    }
    if ticks.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(plotting_error(
            builtin,
            format!(
                "{builtin}: tick values must remain distinct in the graphics coordinate domain"
            ),
        ));
    }
    Ok(ticks)
}

fn numeric_scalar_strictly_less(lo: NumericScalar, hi: NumericScalar) -> bool {
    match (lo, hi) {
        (NumericScalar::F64(lo), NumericScalar::F64(hi)) => lo < hi,
        (NumericScalar::F32(lo), NumericScalar::F32(hi)) => lo < hi,
        (NumericScalar::I8(lo), NumericScalar::I8(hi)) => lo < hi,
        (NumericScalar::I16(lo), NumericScalar::I16(hi)) => lo < hi,
        (NumericScalar::I32(lo), NumericScalar::I32(hi)) => lo < hi,
        (NumericScalar::I64(lo), NumericScalar::I64(hi)) => lo < hi,
        (NumericScalar::U8(lo), NumericScalar::U8(hi)) => lo < hi,
        (NumericScalar::U16(lo), NumericScalar::U16(hi)) => lo < hi,
        (NumericScalar::U32(lo), NumericScalar::U32(hi)) => lo < hi,
        (NumericScalar::U64(lo), NumericScalar::U64(hi)) => lo < hi,
        _ => lo.materialize_f64() < hi.materialize_f64(),
    }
}

fn axis_bounds(bounds: Option<(f64, f64, f64, f64)>, axis: TickAxis) -> Option<(f64, f64)> {
    bounds.map(|(x_min, x_max, y_min, y_max)| match axis {
        TickAxis::X => (x_min, x_max),
        TickAxis::Y => (y_min, y_max),
    })
}

fn ticks_or_auto(explicit: Option<&[f64]>, bounds: Option<(f64, f64)>) -> Vec<f64> {
    if let Some(ticks) = explicit {
        return ticks.to_vec();
    }
    let (lo, hi) = bounds.unwrap_or((-1.0, 1.0));
    runmat_plot::core::plot_utils::generate_major_ticks(lo, hi)
}

fn tick_value(data: Vec<f64>) -> Value {
    let len = data.len();
    Value::Tensor(Tensor::new(data, vec![1, len]).expect("tick row vector"))
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[test]
    fn wide_integer_ticks_compare_exactly_then_reject_graphics_collapse() {
        let base = 1_u64 << 63;
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![base, base + 1]), vec![1, 2])
                .expect("wide ticks"),
        );
        let error = ticks_from_value(&value, "xticks").expect_err("graphics collapse");
        assert!(error.message().contains("graphics coordinate domain"));
    }

    #[test]
    fn integer_ticks_reject_native_ordering_failure() {
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I64(vec![3, 2]), vec![1, 2])
                .expect("ordered ticks"),
        );
        let error = ticks_from_value(&value, "yticks").expect_err("ordering failure");
        assert!(error.message().contains("strictly increasing"));
    }
}
