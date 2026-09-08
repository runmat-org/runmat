mod basic;
mod character_shape;
mod complex_integer;
mod integer;
mod providers;

use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, Value};

use super::implementation;
use crate::BuiltinResult;

fn run(value: Value) -> BuiltinResult<Value> {
    block_on(implementation::execute(value))
}

fn scalar_cell(values: &[f64], rows: usize, columns: usize) -> Value {
    let cells = values.iter().map(|value| Value::Num(*value)).collect();
    crate::make_cell(cells, rows, columns).expect("cell")
}

fn append_same_class(left: &IntegerStorage, right: &IntegerStorage) -> IntegerStorage {
    macro_rules! append {
        ($left:expr, $right:expr, $variant:ident) => {{
            let mut values = $left.clone();
            values.extend_from_slice($right);
            IntegerStorage::$variant(values)
        }};
    }
    match (left, right) {
        (IntegerStorage::I8(left), IntegerStorage::I8(right)) => append!(left, right, I8),
        (IntegerStorage::I16(left), IntegerStorage::I16(right)) => append!(left, right, I16),
        (IntegerStorage::I32(left), IntegerStorage::I32(right)) => append!(left, right, I32),
        (IntegerStorage::I64(left), IntegerStorage::I64(right)) => append!(left, right, I64),
        (IntegerStorage::U8(left), IntegerStorage::U8(right)) => append!(left, right, U8),
        (IntegerStorage::U16(left), IntegerStorage::U16(right)) => append!(left, right, U16),
        (IntegerStorage::U32(left), IntegerStorage::U32(right)) => append!(left, right, U32),
        (IntegerStorage::U64(left), IntegerStorage::U64(right)) => append!(left, right, U64),
        _ => panic!("test inputs must have matching integer classes"),
    }
}

#[test]
fn rejects_non_cell_input() {
    let error = run(Value::Num(1.0)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:cell2mat:InvalidInput"));
}
