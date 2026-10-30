use runmat_value::{CharArray, Value};

pub(super) fn parse(value: &Value) -> crate::BuiltinResult<Vec<String>> {
    let names =
        match value {
            Value::String(runmat_value::StringElement::Text(text)) => vec![text.to_string()],
            Value::StringArray(array) => array
                .data()
                .iter()
                .map(|text| {
                    text.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| super::error::invalid("fields must be text or cellstr"))
                })
                .collect::<crate::BuiltinResult<Vec<_>>>()?,
            Value::CharArray(chars) if chars.rows == 1 => vec![
                runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                    let mut error = super::error::invalid(source.to_string());
                    error.source = Some(Box::new(source));
                    error
                })?,
            ],
            Value::CharArray(chars) => character_rows(chars)?,
            Value::Cell(cell) => cell
                .iter_column_major()
                .map(|value| {
                    crate::builtins::structs::field_name::decode(value)
                        .map_err(|_| super::error::invalid("cell field names must be text scalars"))
                })
                .collect::<crate::BuiltinResult<Vec<_>>>()?,
            _ => return Err(super::error::invalid("fields must be text or cellstr")),
        };
    if names.is_empty() || names.iter().any(String::is_empty) {
        return Err(super::error::invalid("field names must not be empty"));
    }
    Ok(names)
}

fn character_rows(chars: &CharArray) -> crate::BuiltinResult<Vec<String>> {
    (0..chars.rows)
        .map(|row| {
            let start = row * chars.cols;
            Ok(
                runmat_value::decode_utf16_code_units(&chars.data()[start..start + chars.cols])
                    .map_err(|source| {
                        let mut error = super::error::invalid(source.to_string());
                        error.source = Some(Box::new(source));
                        error
                    })?
                    .trim_end()
                    .to_string(),
            )
        })
        .collect()
}
