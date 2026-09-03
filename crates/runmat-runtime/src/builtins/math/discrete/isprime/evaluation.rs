use runmat_value::{IntValue, LogicalArray, NumericStorage, Value};

use super::{isprime_error, IsPrimeError};
use crate::builtins::math::discrete::number_theory::is_prime;
use crate::BuiltinResult;

pub(super) fn evaluate(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => Ok(Value::Bool(test_float(value)?)),
        Value::Int(value) => Ok(Value::Bool(test_int(value)?)),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let storage = tensor
                .into_numeric_storage()
                .map_err(|detail| isprime_error(IsPrimeError::Internal, detail))?;
            let data = test_storage(storage)?;
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(|detail| isprime_error(IsPrimeError::Internal, detail))
        }
        _ => Err(invalid()),
    }
}

fn test_storage(storage: NumericStorage) -> BuiltinResult<Vec<u8>> {
    macro_rules! unsigned {
        ($values:expr) => {
            $values
                .into_iter()
                .map(|value| u8::from(is_prime(u64::from(value))))
                .collect()
        };
    }
    macro_rules! signed {
        ($values:expr) => {{
            let values = $values;
            if values.iter().any(|&value| value < 0) {
                return Err(invalid());
            }
            values
                .into_iter()
                .map(|value| u8::from(is_prime(value as u64)))
                .collect()
        }};
    }
    Ok(match storage {
        NumericStorage::F64(values) => floats(values)?,
        NumericStorage::F32(values) => floats(values.into_iter().map(f64::from))?,
        NumericStorage::I8(values) => signed!(values),
        NumericStorage::I16(values) => signed!(values),
        NumericStorage::I32(values) => signed!(values),
        NumericStorage::I64(values) => signed!(values),
        NumericStorage::U8(values) => unsigned!(values),
        NumericStorage::U16(values) => unsigned!(values),
        NumericStorage::U32(values) => unsigned!(values),
        NumericStorage::U64(values) => unsigned!(values),
    })
}

fn floats(values: impl IntoIterator<Item = f64>) -> BuiltinResult<Vec<u8>> {
    values
        .into_iter()
        .map(test_float)
        .map(|result| result.map(u8::from))
        .collect()
}

fn test_float(value: f64) -> BuiltinResult<bool> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= u64::MAX as f64 {
        return Err(invalid());
    }
    Ok(is_prime(value as u64))
}

fn test_int(value: IntValue) -> BuiltinResult<bool> {
    let value = match value {
        IntValue::I8(value) => u64::try_from(value).ok(),
        IntValue::I16(value) => u64::try_from(value).ok(),
        IntValue::I32(value) => u64::try_from(value).ok(),
        IntValue::I64(value) => u64::try_from(value).ok(),
        IntValue::U8(value) => Some(u64::from(value)),
        IntValue::U16(value) => Some(u64::from(value)),
        IntValue::U32(value) => Some(u64::from(value)),
        IntValue::U64(value) => Some(value),
    }
    .ok_or_else(invalid)?;
    Ok(is_prime(value))
}

fn invalid() -> crate::RuntimeError {
    isprime_error(
        IsPrimeError::InvalidInput,
        "expected real nonnegative integer values",
    )
}
