use crate::builtins::structs::field_name;
use runmat_value::{CharArray, Value};

pub(super) fn parse(value: &Value) -> crate::BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::Cell(array) if !starts_with_structure(array) => array
            .iter_column_major()
            .enumerate()
            .map(|(index, value)| scalar(value, format!("cell element {}", index + 1)))
            .collect::<crate::BuiltinResult<Vec<_>>>()
            .map(Some),
        Value::StringArray(array) => array
            .data
            .iter()
            .enumerate()
            .map(|(index, name)| nonempty(name.clone(), format!("string element {}", index + 1)))
            .collect::<crate::BuiltinResult<Vec<_>>>()
            .map(Some),
        Value::String(_) => scalar(value, "string input").map(|name| Some(vec![name])),
        Value::CharArray(array) => character_rows(array).map(Some),
        _ => Ok(None),
    }
}

fn starts_with_structure(array: &runmat_value::CellArray) -> bool {
    matches!(array.iter_column_major().next(), Some(Value::Struct(_)))
}

fn character_rows(array: &CharArray) -> crate::BuiltinResult<Vec<String>> {
    (0..array.rows)
        .map(|row| {
            let start = row * array.cols;
            let mut name = array.data[start..start + array.cols]
                .iter()
                .collect::<String>();
            while name.ends_with(' ') {
                name.pop();
            }
            nonempty(name, format!("character row {}", row + 1))
        })
        .collect()
}

fn scalar(value: &Value, context: impl std::fmt::Display) -> crate::BuiltinResult<String> {
    let name =
        field_name::decode(value).map_err(|_| super::super::error::invalid_name(&context))?;
    nonempty(name, context)
}

fn nonempty(name: String, context: impl std::fmt::Display) -> crate::BuiltinResult<String> {
    if name.is_empty() {
        return Err(super::super::error::empty_name(context));
    }
    Ok(name)
}
