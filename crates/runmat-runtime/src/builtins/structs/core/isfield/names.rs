use runmat_value::{CellArray, StringArray, StringElement, Value};

use crate::builtins::structs::field_name;

pub(super) enum Query {
    Scalar(StringElement),
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
            crate::builtins::common::text::decode_utf16_with_error(
                array.data(),
                super::error::field_name_type,
            )
            .map(|name| Query::Scalar(StringElement::text(name)))
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
        Err(source) if matches!(value, Value::CharArray(_) | Value::StringArray(_)) => {
            let mut error = super::error::field_name_type();
            if let field_name::ExpectedScalarFieldName::Encoding(source) = source {
                error.source = Some(Box::new(source));
            }
            Err(error)
        }
        Err(_) => Err(super::error::cell_element(value)),
    }
}
