use runmat_plot::plots::figure::{AxisLimitAxis, AxisLimitMethod};
use runmat_value::{NumericScalar, Tensor, Value};

use super::axes_target::{prepare_axes_call, PreparedAxesTarget};
use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::state::ResolvedAxesTarget;
use crate::BuiltinResult;

pub struct PreparedLimitCall {
    pub target: PreparedAxesTarget,
    pub command: LimitCommand,
}

pub fn limit_state_error(
    builtin: &'static str,
    err: impl std::error::Error + Send + Sync + 'static,
) -> crate::RuntimeError {
    crate::builtins::plotting::plotting_error_with_source(builtin, format!("{builtin}: {err}"), err)
}

pub fn prepare_limit_call(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
    color_limits: bool,
) -> BuiltinResult<PreparedLimitCall> {
    let (target, controls) = prepare_limit_controls(runtime, builtin, args)?;
    let command = parse_limit_command(builtin, controls, color_limits)?;
    Ok(PreparedLimitCall { target, command })
}

/// Select the existing compatibility axes target without reading control data
/// or creating/selecting a graphics scene. Family admission uses these exact
/// borrowed controls before invoking the shared limit parser.
pub fn prepare_limit_controls<'a>(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<(PreparedAxesTarget, crate::call::FunctionArgumentView<'a>)> {
    prepare_axes_call(runtime, builtin, args)
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
    args: crate::call::FunctionArgumentView<'_>,
    color_limits: bool,
) -> BuiltinResult<LimitCommand> {
    validate_limit_control_count(builtin, args)?;
    if args.is_empty() {
        return Ok(LimitCommand::Query);
    }
    let arg = &args[0];
    if let Some(text) =
        crate::builtins::plotting::style::value_as_string(arg).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                builtin,
                source.to_string(),
                source,
            )
        })?
    {
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

pub fn validate_limit_control_count(
    builtin: &'static str,
    controls: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if controls.len() > 1 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected at most one argument"),
        ));
    }
    Ok(())
}

pub fn query_limit_method(
    builtin: &'static str,
    axis: AxisLimitAxis,
    target: &ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    let (handle, index) = target.coordinates();
    let property = target.authority();
    let method = crate::builtins::plotting::state::axis_limit_method_for_axes_for_property(
        property, handle, index, axis,
    )
    .map_err(|err| limit_state_error(builtin, err))?;
    Ok(Value::String(method.as_str().into()))
}

pub fn set_limit_method(
    builtin: &'static str,
    axis: AxisLimitAxis,
    method: AxisLimitMethod,
    target: &ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    let (handle, index) = target.coordinates();
    let property = target.authority();
    crate::builtins::plotting::state::set_axis_limit_method_for_axes_for_property(
        property, handle, index, axis, method,
    )
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
