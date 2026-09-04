mod numeric;
mod provider;
mod representations;

use runmat_value::Value;

fn outputs(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values,
        other => panic!("expected output list, got {other:?}"),
    }
}
