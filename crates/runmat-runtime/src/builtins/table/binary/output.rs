use runmat_value::{ObjectInstance, StructValue, Value};

use crate::BuiltinResult;

pub(crate) fn finish(
    source: &ObjectInstance,
    variables: Vec<(String, Value)>,
) -> BuiltinResult<Value> {
    let mut output = StructValue::new();
    for (name, value) in variables {
        output.insert(name, value);
    }
    super::super::table_replace_variables_like(source, output)
}
