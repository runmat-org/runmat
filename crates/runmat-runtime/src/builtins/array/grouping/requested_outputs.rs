use runmat_value::Value;

use crate::BuiltinResult;

pub(crate) fn finish(outputs: Vec<Value>) -> BuiltinResult<Value> {
    if let Some(count) = crate::output_count::current_output_count() {
        if count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            count, outputs,
        ));
    }
    Ok(outputs
        .into_iter()
        .next()
        .unwrap_or(Value::OutputList(Vec::new())))
}
