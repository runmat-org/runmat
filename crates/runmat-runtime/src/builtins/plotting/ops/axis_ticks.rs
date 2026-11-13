use runmat_value::NumericScalar;
use runmat_value::{Tensor, Value};

use super::op_common::axes_target::prepare_axes_call;
use super::state::{
    axis_display_bounds_snapshot_for_axes_for_property, axis_ticks_snapshot_for_axes_for_property,
    set_axis_ticks_for_axes_for_property,
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
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    axis: TickAxis,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let (target, args) = prepare_axes_call(runtime, builtin, args.view())?;
    match (args.len(), args.first()) {
        (0, _) => query_ticks(builtin, axis, &target.resolve(builtin)?),
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
                        apply_mode(builtin, axis, &target.resolve(builtin)?, TickMode::Auto)?;
                        return Ok(Value::String("auto".into()));
                    }
                    "manual" => {
                        super::op_common::axes_output::validate_setter_output(builtin)?;
                        apply_mode(builtin, axis, &target.resolve(builtin)?, TickMode::Manual)?;
                        return Ok(Value::String("manual".into()));
                    }
                    "mode" => return query_mode(builtin, axis, &target.resolve(builtin)?),
                    _ => {}
                }
            }
            let ticks = ticks_from_value(value, builtin)?;
            super::op_common::axes_output::validate_setter_output(builtin)?;
            set_ticks(builtin, axis, &target.resolve(builtin)?, ticks.clone())?;
            Ok(tick_value(ticks))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick argument"),
        )),
    }
}

fn query_ticks(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    let ticks = {
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
        match axis {
            TickAxis::X => ticks_or_auto(x_ticks.as_deref(), axis_bounds(bounds, axis)),
            TickAxis::Y => ticks_or_auto(y_ticks.as_deref(), axis_bounds(bounds, axis)),
        }
    };
    Ok(tick_value(ticks))
}

fn query_mode(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    let manual = {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let (x_ticks, y_ticks) = axis_ticks_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        match axis {
            TickAxis::X => x_ticks.is_some(),
            TickAxis::Y => y_ticks.is_some(),
        }
    };
    Ok(Value::String(if manual { "manual" } else { "auto" }.into()))
}

fn apply_mode(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
    mode: TickMode,
) -> BuiltinResult<()> {
    let current = match mode {
        TickMode::Auto => None,
        TickMode::Manual => {
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
            Some(match axis {
                TickAxis::X => ticks_or_auto(x_ticks.as_deref(), axis_bounds(bounds, axis)),
                TickAxis::Y => ticks_or_auto(y_ticks.as_deref(), axis_bounds(bounds, axis)),
            })
        }
    };
    match current {
        Some(ticks) => set_ticks(builtin, axis, target, ticks),
        None => clear_ticks(builtin, axis, target),
    }
}

fn set_ticks(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
    ticks: Vec<f64>,
) -> BuiltinResult<()> {
    {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let (x_ticks, y_ticks) = axis_ticks_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        let (x, y) = match axis {
            TickAxis::X => (Some(ticks), y_ticks),
            TickAxis::Y => (x_ticks, Some(ticks)),
        };
        set_axis_ticks_for_axes_for_property(property, handle, axes_index, x, y)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))
    }
}

fn clear_ticks(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<()> {
    {
        let (handle, axes_index) = target.coordinates();
        let property = target.authority();

        let (x_ticks, y_ticks) = axis_ticks_snapshot_for_axes_for_property(
            property, handle, axes_index,
        )
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
        let (x, y) = match axis {
            TickAxis::X => (None, y_ticks),
            TickAxis::Y => (x_ticks, None),
        };
        set_axis_ticks_for_axes_for_property(property, handle, axes_index, x, y)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))
    }
}

pub(super) fn tick_tensor_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Tensor> {
    // A scalar numeric Value is a one-element tick vector. Use the canonical
    // checked conversion so native integers keep their exact storage until
    // the graphics boundary. Resident controls remain unsupported.
    match value {
        Value::Num(_) | Value::Int(_) => tensor::value_into_tensor_for(builtin, value.clone()),
        _ => Tensor::try_from(value),
    }
    .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))
}

fn ticks_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    let tensor = tick_tensor_from_value(value, builtin)?;
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

fn value_as_string(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::CharArray(chars) => Some(runmat_value::decode_utf16_code_units(chars.data())?),
        _ => None,
    })
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
