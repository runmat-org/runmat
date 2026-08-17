use runmat_builtins::{NumericScalar, Tensor, Value};

use crate::builtins::plotting::plotting_error;
use crate::BuiltinResult;

#[derive(Clone, Debug)]
pub enum LimitCommand {
    Query,
    Set(Option<(f64, f64)>),
}

pub fn parse_limit_command(builtin: &'static str, args: &[Value]) -> BuiltinResult<LimitCommand> {
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
            "auto" | "tight" => Ok(LimitCommand::Set(None)),
            "manual" => Ok(LimitCommand::Query),
            _ => Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported mode `{normalized}`"),
            )),
        };
    }
    let limits = limits_from_value(arg, builtin)?;
    Ok(LimitCommand::Set(Some(limits)))
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
    use runmat_builtins::IntegerStorage;

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
