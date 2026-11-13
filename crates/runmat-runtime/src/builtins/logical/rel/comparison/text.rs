//! Text-carrier normalization for relational comparison.
use super::{runtime_error, ComparisonError, RelationalOperator, Value};

pub(super) fn normalize_text(
    lhs: Value,
    rhs: Value,
    operator: RelationalOperator,
) -> crate::BuiltinResult<(Value, Value)> {
    Ok(match (lhs, rhs) {
        (Value::CharArray(chars), Value::String(text)) => (
            Value::string(runmat_value::decode_utf16_code_units(chars.data()).map_err(
                |source| {
                    let mut error = runtime_error(operator, ComparisonError::InvalidInput);
                    error.source = Some(Box::new(source));
                    error
                },
            )?),
            Value::String(text),
        ),
        (Value::String(text), Value::CharArray(chars)) => (
            Value::String(text),
            Value::string(runmat_value::decode_utf16_code_units(chars.data()).map_err(
                |source| {
                    let mut error = runtime_error(operator, ComparisonError::InvalidInput);
                    error.source = Some(Box::new(source));
                    error
                },
            )?),
        ),
        (Value::CharArray(chars), Value::StringArray(text)) => (
            Value::string(runmat_value::decode_utf16_code_units(chars.data()).map_err(
                |source| {
                    let mut error = runtime_error(operator, ComparisonError::InvalidInput);
                    error.source = Some(Box::new(source));
                    error
                },
            )?),
            Value::StringArray(text),
        ),
        (Value::StringArray(text), Value::CharArray(chars)) => (
            Value::StringArray(text),
            Value::string(runmat_value::decode_utf16_code_units(chars.data()).map_err(
                |source| {
                    let mut error = runtime_error(operator, ComparisonError::InvalidInput);
                    error.source = Some(Box::new(source));
                    error
                },
            )?),
        ),
        values => values,
    })
}
