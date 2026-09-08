use runmat_value::{CellArray, Value};

pub(super) enum Query {
    Scalar(String),
    Collection {
        names: Vec<String>,
        shape: Vec<usize>,
    },
}

pub(super) fn parse(value: Value) -> crate::BuiltinResult<Query> {
    match value {
        Value::String(name) => Ok(Query::Scalar(name)),
        Value::CharArray(array) if array.rows == 1 => {
            Ok(Query::Scalar(array.data.iter().collect()))
        }
        Value::CharArray(_) => Err(super::error::field_name_type()),
        Value::StringArray(array) => Ok(Query::Collection {
            names: array.data,
            shape: array.shape,
        }),
        Value::Cell(array) => Ok(Query::Collection {
            names: cell_names(&array)?,
            shape: query_shape(&array),
        }),
        _ => Err(super::error::field_name_type()),
    }
}

fn query_shape(array: &CellArray) -> Vec<usize> {
    if array.shape.is_empty() {
        vec![array.rows, array.cols]
    } else {
        array.shape.clone()
    }
}

fn cell_names(array: &CellArray) -> crate::BuiltinResult<Vec<String>> {
    array.to_column_major().iter().map(cell_name).collect()
}

fn cell_name(value: &Value) -> crate::BuiltinResult<String> {
    match value {
        Value::String(name) => Ok(name.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(_) | Value::StringArray(_) => Err(super::error::field_name_type()),
        other => Err(super::error::cell_element(other)),
    }
}
