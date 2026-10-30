use runmat_value::{CellArray, CharArray, Value};

pub(super) fn convert(array: CellArray) -> crate::BuiltinResult<Value> {
    let shape = array.shape.as_slice().to_vec();
    let values = array
        .data()
        .iter()
        .map(character)
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    CellArray::new_with_shape(values, shape)
        .map(Value::Cell)
        .map_err(super::super::error::internal)
}

fn character(value: &Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::CharArray(array) if is_vector(array) => Ok(Value::CharArray(array.clone())),
        Value::String(text) => Ok(Value::CharArray(CharArray::new_row(
            text.as_str()
                .ok_or_else(super::super::error::invalid_contents)?,
        ))),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(Value::CharArray(CharArray::new_row(
                array.data()[0]
                    .as_str()
                    .ok_or_else(super::super::error::invalid_contents)?,
            )))
        }
        _ => Err(super::super::error::invalid_contents()),
    }
}

fn is_vector(array: &CharArray) -> bool {
    array.rows == 1 || (array.rows == 0 && array.cols == 0)
}
