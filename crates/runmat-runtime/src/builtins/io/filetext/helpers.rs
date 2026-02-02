use runmat_builtins::{CharArray, Value};

pub(crate) fn extract_scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

pub(crate) fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

pub(crate) fn normalize_encoding_label(label: &str) -> String {
    label.trim().to_ascii_lowercase()
}
