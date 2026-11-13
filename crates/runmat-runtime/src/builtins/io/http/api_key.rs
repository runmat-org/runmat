//! HTTP API-key metadata from a leading options structure.

use http::header::{HeaderName, HeaderValue, InvalidHeaderName, InvalidHeaderValue};
use runmat_value::{NumericScalar, StructValue, Value};
use std::fmt;

#[derive(Debug)]
pub(super) enum ApiKeyError {
    InvalidName,
    InvalidValue,
    MissingName,
    Encoding(runmat_value::Utf16DecodeError),
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
            Self::Encoding(_) => formatter.write_str("API key text contains malformed UTF-16"),
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
            Self::Encoding(source) => Some(source),
            _ => None,
        }
    }
}

/// Direct `webread`/`webwrite` name-value pairs remain query/post names. Only
/// the leading options structure carries these weboptions-specific fields.
#[derive(Debug, Default)]
pub(super) struct PreparedApiKey {
    name_field: Option<String>,
    value_field: Option<String>,
    field_names: Vec<String>,
}

impl PreparedApiKey {
    pub(super) fn parse(source: &StructValue) -> Self {
        let mut prepared = Self::default();
        for name in source.fields.keys() {
            match runmat_builtins::WebOptionField::decode(name) {
                Some(runmat_builtins::WebOptionField::KeyName) => {
                    prepared.name_field = Some(name.to_owned());
                    prepared.field_names.push(name.to_owned());
                }
                Some(runmat_builtins::WebOptionField::KeyValue) => {
                    prepared.value_field = Some(name.to_owned());
                    prepared.field_names.push(name.to_owned());
                }
                _ => {}
            }
        }
        // BTreeMap traversal and assignment retain the last spelling when a
        // struct contains two differently cased aliases of the same field.
        prepared
    }

    pub(super) fn contains_field(&self, name: &str) -> bool {
        self.field_names.iter().any(|field| field == name)
    }

    pub(super) fn preflight_known(&self, source: &StructValue) -> Result<(), ApiKeyError> {
        let (name, value) = self.components(source, true)?;
        validate_components(name, value).map(|_| ())
    }

    pub(super) fn finish(
        &self,
        source: &StructValue,
    ) -> Result<Option<(String, String)>, ApiKeyError> {
        let (name, value) = self.components(source, false)?;
        Ok(validate_components(name, value)?.flatten())
    }

    fn components(
        &self,
        source: &StructValue,
        skip_resident: bool,
    ) -> Result<(Option<String>, Option<String>), ApiKeyError> {
        let name = match self
            .name_field
            .as_ref()
            .and_then(|field| source.fields.get(field))
        {
            Some(value) if skip_resident && crate::dispatcher::value_contains_gpu(value) => None,
            Some(value) => Some(text_scalar(value)?.ok_or(ApiKeyError::InvalidName)?),
            None => Some(String::new()),
        };
        let value = match self
            .value_field
            .as_ref()
            .and_then(|field| source.fields.get(field))
        {
            Some(value) if skip_resident && crate::dispatcher::value_contains_gpu(value) => None,
            Some(value) => Some(key_value(value)?.ok_or(ApiKeyError::InvalidValue)?),
            None => Some(String::new()),
        };
        Ok((name, value))
    }
}

fn validate_components(
    name: Option<String>,
    value: Option<String>,
) -> Result<Option<Option<(String, String)>>, ApiKeyError> {
    if let Some(name) = name.as_deref().filter(|name| !name.trim().is_empty()) {
        HeaderName::from_bytes(name.as_bytes()).map_err(ApiKeyError::InvalidHeaderName)?;
    }
    if let Some(value) = value.as_deref() {
        HeaderValue::from_str(value).map_err(ApiKeyError::InvalidHeaderValue)?;
    }
    match (name, value) {
        (Some(name), Some(value)) if name.trim().is_empty() && !value.is_empty() => {
            Err(ApiKeyError::MissingName)
        }
        (Some(name), Some(_)) if name.trim().is_empty() => Ok(Some(None)),
        (Some(name), Some(value)) => Ok(Some(Some((name, value)))),
        _ => Ok(None),
    }
}

fn text_scalar(value: &Value) -> Result<Option<String>, ApiKeyError> {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => {
            Ok(Some(value.as_str().to_owned()))
        }
        Value::CharArray(value) if value.rows == 1 => {
            runmat_value::decode_utf16_code_units(value.data())
                .map(Some)
                .map_err(ApiKeyError::Encoding)
        }
        Value::StringArray(value) if value.data().len() == 1 && !value.data()[0].is_missing() => {
            Ok(Some(
                value.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

fn key_value(value: &Value) -> Result<Option<String>, ApiKeyError> {
    match value {
        Value::Num(value) => Ok(Some(value.to_string())),
        Value::Int(value) => Ok(Some(value.decimal_string())),
        Value::Bool(value) => Ok(Some(value.to_string())),
        Value::Tensor(value) if crate::builtins::common::tensor::is_scalar_tensor(value) => {
            Ok(value.numeric_value_at(0).map(numeric_scalar))
        }
        Value::LogicalArray(value) if value.len() == 1 => {
            Ok(Some((value.data[0] != 0).to_string()))
        }
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
        let mut source = runmat_value::UnboundedStructValueBuilder::new();
        source
            .fields
            .insert("KeyName".into(), Value::from("X-Sequence"));
        source
            .fields
            .insert("KeyValue".into(), Value::Int(IntValue::U64(u64::MAX)));
        let mut source = source.finish();
        assert_eq!(
            PreparedApiKey::parse(&source).finish(&source).unwrap(),
            Some(("X-Sequence".into(), u64::MAX.to_string()))
        );
        source
            .try_replace_unique("KeyName", Value::from(""))
            .expect("unique field");
        source
            .try_replace_unique("KeyValue", Value::from(""))
            .expect("unique field");
        assert_eq!(
            PreparedApiKey::parse(&source).finish(&source).unwrap(),
            None
        );
        source
            .try_replace_unique("KeyValue", Value::from("secret"))
            .expect("unique field");
        assert!(matches!(
            PreparedApiKey::parse(&source).finish(&source),
            Err(ApiKeyError::MissingName)
        ));
    }

    #[test]
    fn independently_known_name_is_validated_before_resident_value_and_case_alias_is_last_wins() {
        let mut source = runmat_value::UnboundedStructValueBuilder::new();
        source
            .fields
            .insert("KeyName".into(), Value::from("bad\nname"));
        source
            .fields
            .insert("KeyValue".into(), Value::from("value"));
        let source = source.finish();
        let prepared = PreparedApiKey::parse(&source);
        assert!(matches!(
            prepared.preflight_known(&source),
            Err(ApiKeyError::InvalidHeaderName(_))
        ));

        let mut source = runmat_value::UnboundedStructValueBuilder::from_structure(source);
        source
            .fields
            .insert("KEYNAME".into(), Value::from("X-First"));
        source
            .fields
            .insert("keyname".into(), Value::from("X-Last"));
        let source = source.finish();
        let prepared = PreparedApiKey::parse(&source);
        assert_eq!(
            prepared.finish(&source).unwrap(),
            Some(("X-Last".into(), "value".into()))
        );
        assert!(prepared.contains_field("KEYNAME"));
        assert!(prepared.contains_field("KeyName"));
    }
}
