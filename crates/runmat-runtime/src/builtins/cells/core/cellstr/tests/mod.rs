mod admission;
mod character;
mod extensions;
mod string;

use runmat_value::{CellArray, Value};

fn call(value: Value) -> crate::BuiltinResult<Value> {
    super::cellstr_builtin(value)
}

fn strings(cell: &CellArray) -> Vec<String> {
    cell.data.iter().map(text).collect()
}

fn text(value: &Value) -> String {
    match value {
        Value::CharArray(array) => array.data.iter().collect(),
        other => panic!("expected character array, found {other:?}"),
    }
}

fn cell(value: Value) -> CellArray {
    let Value::Cell(cell) = value else {
        panic!("expected cell output");
    };
    cell
}
