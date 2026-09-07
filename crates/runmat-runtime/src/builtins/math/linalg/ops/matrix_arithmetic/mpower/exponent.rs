use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

pub(super) fn parse(value: &Value) -> BuiltinResult<Option<i32>> {
    match value {
        Value::Int(value) => parse_exact(value.try_to_i64()).map(Some),
        Value::Num(value) => parse_floating(*value).map(Some),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            if let Some(storage) = value.integer_storage() {
                let value = storage
                    .value_at(0)
                    .expect("scalar integer tensor has one stored value");
                return parse_exact(value.try_to_i64()).map(Some);
            }
            parse_floating(tensor::tensor_value_f64(value, 0)).map(Some)
        }
        _ => Ok(None),
    }
}

fn parse_exact(value: Option<i64>) -> BuiltinResult<i32> {
    value
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(exponent_range_error)
}

fn parse_floating(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(super::errors::invalid_argument(
            runmat_builtins::MPOWER_ERROR_INVALID_ARGUMENT.message,
        ));
    }
    if value < i32::MIN as f64 || value > i32::MAX as f64 {
        return Err(exponent_range_error());
    }
    Ok(value as i32)
}

fn exponent_range_error() -> crate::RuntimeError {
    super::errors::invalid_argument(
        "mpower: exponent magnitude exceeds supported range (|n| ≤ 2^31−1)",
    )
}
