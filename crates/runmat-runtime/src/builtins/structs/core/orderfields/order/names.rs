use crate::builtins::structs::field_name;
use runmat_value::{CharArray, Value};

pub(super) fn parse(value: &Value) -> crate::BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::Cell(array) => array
            .iter_column_major()
            .enumerate()
            .map(|(index, value)| scalar(value, format!("cell element {}", index + 1)))
            .collect::<crate::BuiltinResult<Vec<_>>>()
            .map(Some),
        Value::StringArray(array) => array
            .data()
            .iter()
            .enumerate()
            .map(|(index, name)| {
                nonempty(
                    name.as_str()
                        .ok_or_else(|| {
                            super::super::error::invalid_name(format!(
                                "string element {}",
                                index + 1
                            ))
                        })?
                        .to_owned(),
                    format!("string element {}", index + 1),
                )
            })
            .collect::<crate::BuiltinResult<Vec<_>>>()
            .map(Some),
        Value::String(_) => scalar(value, "string input").map(|name| Some(vec![name])),
        Value::CharArray(array) => character_rows(array).map(Some),
        _ => Ok(None),
    }
}

fn character_rows(array: &CharArray) -> crate::BuiltinResult<Vec<String>> {
    (0..array.rows)
        .map(|row| {
            let start = row * array.cols;
            let mut name = crate::builtins::common::text::decode_utf16_with_error(
                &array.data()[start..start + array.cols],
                || super::super::error::invalid_name(format!("character row {}", row + 1)),
            )?;
            while name.ends_with(' ') {
                name.pop();
            }
            nonempty(name, format!("character row {}", row + 1))
        })
        .collect()
}

fn scalar(value: &Value, context: impl std::fmt::Display) -> crate::BuiltinResult<String> {
    let name = field_name::decode(value).map_err(|source| {
        let mut error = super::super::error::invalid_name(&context);
        if let field_name::ExpectedScalarFieldName::Encoding(source) = source {
            error.source = Some(Box::new(source));
        }
        error
    })?;
    nonempty(name, context)
}

fn nonempty(name: String, context: impl std::fmt::Display) -> crate::BuiltinResult<String> {
    if name.is_empty() {
        return Err(super::super::error::empty_name(context));
    }
    Ok(name)
}
