mod compatibility;
mod names;
mod structures;
mod validation;

use runmat_value::Value;

pub(super) fn run(target: Value, fields: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::rmfield_builtin(target, fields))
}

pub(super) fn structure(fields: &[(&str, Value)]) -> Value {
    let mut structure = runmat_value::StructValue::new();
    for (name, value) in fields {
        structure.fields.insert((*name).into(), value.clone());
    }
    Value::Struct(structure)
}
