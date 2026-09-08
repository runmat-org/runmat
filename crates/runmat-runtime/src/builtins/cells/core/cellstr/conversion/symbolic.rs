use runmat_value::{CellArray, CharArray, SymbolicArray, SymbolicExpr, Value};

pub(super) fn scalar(expression: SymbolicExpr) -> crate::BuiltinResult<Value> {
    CellArray::new(vec![character(expression)], 1, 1)
        .map(Value::Cell)
        .map_err(super::super::error::internal)
}

pub(super) fn array(array: SymbolicArray) -> crate::BuiltinResult<Value> {
    let values = array.data.into_iter().map(character).collect();
    CellArray::from_column_major(values, array.shape)
        .map(Value::Cell)
        .map_err(super::super::error::internal)
}

fn character(expression: SymbolicExpr) -> Value {
    Value::CharArray(CharArray::new_row(&expression.to_string()))
}
