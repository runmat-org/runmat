use runmat_plot::plots::figure::{AxisLimitAxis, AxisLimitMethod};
use runmat_value::{NumericScalar, Tensor, Value};

use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::FigureHandle;
use crate::BuiltinResult;

pub type LimitTarget = Option<(FigureHandle, usize)>;

pub struct PreparedLimitCall {
    pub target: LimitTarget,
    pub command: LimitCommand,
}

pub fn selected_limit_axes(target: LimitTarget) -> (FigureHandle, usize) {
    target.unwrap_or_else(|| {
        let current = crate::builtins::plotting::state::current_axes_state();
        (current.handle, current.active_index)
    })
}

pub fn limit_state_error(
    builtin: &'static str,
    err: impl std::error::Error + Send + Sync + 'static,
) -> crate::RuntimeError {
    crate::builtins::plotting::plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
}

pub fn prepare_limit_call(
    builtin: &'static str,
    args: &[Value],
    color_limits: bool,
) -> BuiltinResult<PreparedLimitCall> {
    let (target, controls) = match args {
        [first, rest @ ..] => match resolve_plot_handle(first, builtin) {
            Ok(PlotHandle::Axes(handle, index)) => (Some((handle, index)), rest),
            _ => (None, args),
        },
        [] => (None, args),
    };
    let command = parse_limit_command(builtin, controls, color_limits)?;
    Ok(PreparedLimitCall { target, command })
}

#[derive(Clone, Debug)]
pub enum LimitCommand {
    Query,
    QueryMode,
    QueryMethod,
    Freeze,
    SetMethod(AxisLimitMethod),
    Set(Option<(f64, f64)>),
}

pub fn parse_limit_command(
    builtin: &'static str,
    args: &[Value],
    color_limits: bool,
) -> BuiltinResult<LimitCommand> {
    if args.is_empty() {
        return Ok(LimitCommand::Query);
    }
    if args.len() > 1 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected at most one argument"),
        ));
    }
    let arg = &args[0];
    if let Some(text) = crate::builtins::plotting::style::value_as_string(arg) {
        let normalized = text.trim().to_ascii_lowercase();
        return match normalized.as_str() {
            "auto" => Ok(LimitCommand::Set(None)),
            "tight" if !color_limits => Ok(LimitCommand::SetMethod(AxisLimitMethod::Tight)),
            "padded" if !color_limits => Ok(LimitCommand::SetMethod(AxisLimitMethod::Padded)),
            "tickaligned" if !color_limits => {
                Ok(LimitCommand::SetMethod(AxisLimitMethod::TickAligned))
            }
            "manual" => Ok(LimitCommand::Freeze),
            "mode" if !color_limits => Ok(LimitCommand::QueryMode),
            "method" if !color_limits => Ok(LimitCommand::QueryMethod),
            _ => Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported mode `{normalized}`"),
            )),
        };
    }
    let limits = limits_from_value(arg, builtin)?;
    Ok(LimitCommand::Set(Some(limits)))
}

pub fn query_limit_method(
    builtin: &'static str,
    axis: AxisLimitAxis,
    target: LimitTarget,
) -> BuiltinResult<Value> {
    let (handle, index) = selected_limit_axes(target);
    let method = crate::builtins::plotting::state::axis_limit_method_for_axes(handle, index, axis)
        .map_err(|err| limit_state_error(builtin, err))?;
    Ok(Value::String(method.as_str().into()))
}

pub fn set_limit_method(
    builtin: &'static str,
    axis: AxisLimitAxis,
    method: AxisLimitMethod,
    target: LimitTarget,
) -> BuiltinResult<Value> {
    let (handle, index) = selected_limit_axes(target);
    crate::builtins::plotting::state::set_axis_limit_method_for_axes(handle, index, axis, method)
        .map_err(|err| limit_state_error(builtin, err))?;
    Ok(Value::String(method.as_str().into()))
}

pub fn limits_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<(f64, f64)> {
    let tensor =
        Tensor::try_from(value).map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
    if tensor.len() != 2 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected a 2-element numeric vector"),
        ));
    }
    let lo_exact = tensor.numeric_value_at(0).expect("validated limit storage");
    let hi_exact = tensor.numeric_value_at(1).expect("validated limit storage");
    let lo = lo_exact.materialize_f64();
    let hi = hi_exact.materialize_f64();
    if !lo.is_finite() || !hi.is_finite() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: limits must be finite"),
        ));
    }
    if !numeric_scalar_strictly_less(lo_exact, hi_exact) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: lower limit must be less than upper limit"),
        ));
    }
    if lo >= hi {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: limits must remain distinct in the graphics coordinate domain"),
        ));
    }
    Ok((lo, hi))
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

pub fn limit_value(limits: Option<(f64, f64)>) -> Value {
    let data = match limits {
        Some((lo, hi)) => vec![lo, hi],
        None => vec![f64::NAN, f64::NAN],
    };
    Value::Tensor(Tensor::new(data, vec![1, 2]).expect("limit vector shape"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[test]
    fn limits_read_typed_integer_storage() {
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![1, 2]).expect("limits"),
        );

        assert_eq!(
            limits_from_value(&value, "xlim").expect("limits"),
            (10.0, 20.0)
        );
    }
}
