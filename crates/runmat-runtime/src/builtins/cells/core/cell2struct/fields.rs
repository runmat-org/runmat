use runmat_value::{CharArray, Value};

pub(super) fn parse(value: &Value) -> crate::BuiltinResult<Vec<String>> {
    let names = match value {
        Value::String(text) => vec![text.clone()],
        Value::StringArray(array) => array.data.clone(),
        Value::CharArray(chars) if chars.rows == 1 => vec![chars.data.iter().collect()],
        Value::CharArray(chars) => character_rows(chars),
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

fn character_rows(chars: &CharArray) -> Vec<String> {
    (0..chars.rows)
        .map(|row| {
            let start = row * chars.cols;
            chars.data[start..start + chars.cols]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}
