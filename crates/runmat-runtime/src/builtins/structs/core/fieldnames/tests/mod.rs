mod objects;
mod structures;
mod support;
mod validation;

use runmat_value::{CellArray, Value};

pub(super) fn run(value: Value) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::fieldnames_builtin(value))
}

pub(super) fn strings(value: Value) -> (Vec<String>, Vec<usize>) {
    let Value::Cell(cell) = value else {
        panic!("expected cell array result");
    };
    (cell_strings(&cell), cell.shape.to_vec())
}

fn cell_strings(cell: &CellArray) -> Vec<String> {
    cell.data()
        .iter()
        .map(|value| match value {
            Value::CharArray(array) => {
                runmat_value::decode_utf16_code_units(array.data()).expect("valid UTF-16 fixture")
            }
            other => panic!("expected character array cell element, got {other:?}"),
        })
        .collect()
}
