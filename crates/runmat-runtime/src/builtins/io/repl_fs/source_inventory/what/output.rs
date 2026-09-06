use runmat_value::{CellArray, CharArray, StructValue, Value};

use crate::builtins::common::fs::path_to_string;
use crate::BuiltinResult;

pub(super) fn value(inventory: super::contents::Inventory) -> BuiltinResult<Value> {
    let mut result = StructValue::new();
    result.insert("path", character(&path_to_string(&inventory.folder)));
    result.insert("m", names(inventory.sources)?);
    result.insert("mat", names(inventory.data)?);
    result.insert("mex", names(inventory.extensions)?);
    result.insert("classes", names(inventory.classes)?);
    result.insert("packages", names(inventory.packages)?);
    Ok(Value::Struct(result))
}

fn names(values: Vec<String>) -> BuiltinResult<Value> {
    let rows = values.len();
    let values = values.into_iter().map(|value| character(&value)).collect();
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(|error| super::error::message(&runmat_builtins::WHAT_ERROR_FILESYSTEM, error))
}

fn character(value: &str) -> Value {
    Value::CharArray(CharArray::new_row(value))
}
