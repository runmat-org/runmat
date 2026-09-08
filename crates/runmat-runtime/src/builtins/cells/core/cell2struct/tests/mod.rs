mod assembly;
mod dimension;
mod fields;
mod preservation;

use runmat_value::{CellArray, StructValue, Value};

fn call(cells: CellArray, fields: Value, dimension: Option<Value>) -> crate::BuiltinResult<Value> {
    super::cell2struct_builtin(Value::Cell(cells), fields, dimension.into_iter().collect())
}

fn field<'a>(structure: &'a StructValue, name: &str) -> &'a Value {
    structure.fields.get(name).expect("test field")
}
