use runmat_value::Value;

pub(super) fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(characters) if characters.rows == 1 => {
            Some(characters.data.iter().collect())
        }
        _ => None,
    }
}
