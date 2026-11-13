use runmat_value::{LogicalArray, Value};
use std::collections::HashSet;

use super::names::Query;

pub(super) fn evaluate(
    query: Query,
    fields: Option<&HashSet<&str>>,
) -> crate::BuiltinResult<Value> {
    match query {
        Query::Scalar(name) => Ok(Value::Bool(
            name.as_str().is_some_and(|name| contains(fields, name)),
        )),
        Query::Strings(array) => {
            let values = array
                .data()
                .iter()
                .map(|name| u8::from(name.as_str().is_some_and(|name| contains(fields, name))))
                .collect();
            LogicalArray::new(values, array.shape.as_slice().to_vec())
                .map(Value::LogicalArray)
                .map_err(|error| super::error::internal(format!("isfield: {error}")))
        }
        Query::Collection { names, shape } => {
            let values = names
                .iter()
                .map(|name| u8::from(contains(fields, name)))
                .collect();
            LogicalArray::new(values, shape)
                .map(Value::LogicalArray)
                .map_err(|error| super::error::internal(format!("isfield: {error}")))
        }
    }
}

fn contains(fields: Option<&HashSet<&str>>, name: &str) -> bool {
    fields.is_some_and(|fields| fields.contains(name))
}
