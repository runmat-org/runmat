//! HTTP API-key metadata from a leading options structure.

use reqwest::header::{HeaderName, HeaderValue, InvalidHeaderName, InvalidHeaderValue};
use runmat_value::{NumericScalar, StructValue, Value};
use std::fmt;

#[derive(Debug)]
pub(super) enum ApiKeyError {
    InvalidName,
    InvalidValue,
    MissingName,
    InvalidHeaderName(InvalidHeaderName),
    InvalidHeaderValue(InvalidHeaderValue),
}

impl fmt::Display for ApiKeyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName => {
                formatter.write_str("KeyName must be a character row or string scalar")
            }
            Self::InvalidValue => {
                formatter.write_str("KeyValue must be a text, numeric, or logical scalar")
            }
            Self::MissingName => formatter.write_str("KeyValue requires a nonempty KeyName"),
            Self::InvalidHeaderName(_) => {
                formatter.write_str("KeyName is not a valid HTTP header name")
            }
            Self::InvalidHeaderValue(_) => {
                formatter.write_str("KeyValue is not a valid HTTP header value")
            }
        }
    }
}

impl std::error::Error for ApiKeyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidHeaderName(source) => Some(source),
            Self::InvalidHeaderValue(source) => Some(source),
            _ => None,
        }
    }
}

/// Direct `webread`/`webwrite` name-value pairs remain query/post names. Only
/// the leading options structure carries these weboptions-specific fields.
pub(super) fn from_options(source: &StructValue) -> Result<Option<(String, String)>, ApiKeyError> {
    let name = source
        .fields
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("KeyName"));
    let value = source
        .fields
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("KeyValue"));
    if name.is_none() && value.is_none() {
        return Ok(None);
    }
    let name = name
        .map(|(_, value)| text_scalar(value).ok_or(ApiKeyError::InvalidName))
        .transpose()?
        .unwrap_or_default();
    let value = value
        .map(|(_, value)| key_value(value).ok_or(ApiKeyError::InvalidValue))
        .transpose()?
        .unwrap_or_default();
    if name.trim().is_empty() {
        return if value.is_empty() {
            Ok(None)
        } else {
            Err(ApiKeyError::MissingName)
        };
    }
    HeaderName::from_bytes(name.as_bytes()).map_err(ApiKeyError::InvalidHeaderName)?;
    HeaderValue::from_str(&value).map_err(ApiKeyError::InvalidHeaderValue)?;
    Ok(Some((name, value)))
}

fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::CharArray(value) if value.rows == 1 => Some(value.data.iter().collect()),
        Value::StringArray(value) if value.data.len() == 1 => Some(value.data[0].clone()),
        _ => None,
    }
}

fn key_value(value: &Value) -> Option<String> {
    match value {
        Value::Num(value) => Some(value.to_string()),
        Value::Int(value) => Some(value.decimal_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Tensor(value) if crate::builtins::common::tensor::is_scalar_tensor(value) => {
            value.numeric_value_at(0).map(numeric_scalar)
        }
        Value::LogicalArray(value) if value.len() == 1 => Some((value.data[0] != 0).to_string()),
        _ => text_scalar(value),
    }
}

fn numeric_scalar(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => value.to_string(),
        NumericScalar::F32(value) => value.to_string(),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntValue;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exact_integer_header_and_empty_defaults() {
        let mut source = StructValue::new();
        source
            .fields
            .insert("KeyName".into(), Value::from("X-Sequence"));
        source
            .fields
            .insert("KeyValue".into(), Value::Int(IntValue::U64(u64::MAX)));
        assert_eq!(
            from_options(&source).unwrap(),
            Some(("X-Sequence".into(), u64::MAX.to_string()))
        );
        source.fields.insert("KeyName".into(), Value::from(""));
        source.fields.insert("KeyValue".into(), Value::from(""));
        assert_eq!(from_options(&source).unwrap(), None);
        source
            .fields
            .insert("KeyValue".into(), Value::from("secret"));
        assert!(matches!(
            from_options(&source),
            Err(ApiKeyError::MissingName)
        ));
    }
}
