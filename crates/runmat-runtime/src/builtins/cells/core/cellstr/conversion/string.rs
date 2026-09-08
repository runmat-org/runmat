use runmat_value::{CellArray, CharArray, StringArray, Value};

pub(super) fn scalar(text: String) -> crate::BuiltinResult<Value> {
    CellArray::new(vec![character(text)], 1, 1)
        .map(Value::Cell)
        .map_err(super::super::error::internal)
}

pub(super) fn array(array: StringArray) -> crate::BuiltinResult<Value> {
    let values = array.data.into_iter().map(character).collect();
    CellArray::from_column_major(values, array.shape)
        .map(Value::Cell)
        .map_err(super::super::error::internal)
}

fn character(text: String) -> Value {
    Value::CharArray(CharArray::new_row(&text))
}
