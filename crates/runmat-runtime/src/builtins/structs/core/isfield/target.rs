use runmat_value::Value;
use std::collections::HashSet;

pub(super) fn common_fields(value: &Value) -> Option<HashSet<&str>> {
    match value {
        Value::Struct(structure) => Some(structure.fields.keys().map(String::as_str).collect()),
        Value::StructArray(array) => Some(array.field_names().map(String::as_str).collect()),
        _ => None,
    }
}
