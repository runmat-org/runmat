use runmat_value::{CharArray, Value};

pub(super) fn field_name_cell(names: Vec<String>) -> crate::BuiltinResult<Value> {
    let rows = names.len();
    let cells = names
        .into_iter()
        .map(|name| Value::CharArray(CharArray::new_row(&name)))
        .collect();
    crate::make_cell(cells, rows, 1)
        .map_err(|error| super::error::internal(format!("fieldnames: {error}")))
}
