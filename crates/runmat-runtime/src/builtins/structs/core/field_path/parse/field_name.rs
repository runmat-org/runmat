//! Scalar public field names, separate from numeric selector parsing.
use super::{error, PathError, PathErrorKind, Value};

pub(super) fn parse_field_name(value: Value) -> Result<String, PathError> {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => Ok(value.to_string()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .as_str()
                .expect("nonmissing scalar text")
                .to_owned())
        }
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data()).map_err(|source| PathError {
                kind: PathErrorKind::FieldName,
                detail: "field name is not valid UTF-16".into(),
                source: Some(source),
            })
        }
        _ => Err(error(
            PathErrorKind::FieldName,
            "field names must be scalar strings or character row vectors",
        )),
    }
}
