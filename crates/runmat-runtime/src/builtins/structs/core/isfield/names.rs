use runmat_value::{CellArray, StringArray, Utf8Text, Value};

use crate::builtins::structs::field_name;

pub(super) enum Query {
    Scalar(Utf8Text),
    Strings(StringArray),
    Collection {
        names: Vec<String>,
        shape: Vec<usize>,
    },
}

pub(super) fn parse(value: Value) -> crate::BuiltinResult<Query> {
    match value {
        Value::String(name) => Ok(Query::Scalar(name)),
        Value::CharArray(array) if array.rows == 1 => {
            Ok(Query::Scalar(Utf8Text::new(array.data.iter().collect())))
        }
        Value::CharArray(_) => Err(super::error::field_name_type()),
        Value::StringArray(array) => Ok(Query::Strings(array)),
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
        array.shape.as_slice().to_vec()
    }
}

fn cell_names(array: &CellArray) -> crate::BuiltinResult<Vec<String>> {
    array.iter_column_major().map(cell_name).collect()
}

fn cell_name(value: &Value) -> crate::BuiltinResult<String> {
    match field_name::decode(value) {
        Ok(name) => Ok(name),
        Err(_) if matches!(value, Value::CharArray(_) | Value::StringArray(_)) => {
            Err(super::error::field_name_type())
        }
        Err(_) => Err(super::error::cell_element(value)),
    }
}
