use runmat_value::{StructValue, Value};

pub(super) fn parse(value: &Value) -> crate::BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::Struct(structure) => Ok(Some(field_order(structure))),
        Value::StructArray(array) => Ok(Some(array.field_names().cloned().collect())),
        _ => Ok(None),
    }
}

fn field_order(structure: &StructValue) -> Vec<String> {
    structure.field_names().cloned().collect()
}
