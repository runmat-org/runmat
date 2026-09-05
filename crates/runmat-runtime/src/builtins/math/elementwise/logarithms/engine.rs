use super::operation::LogarithmOperation;
use crate::builtins::math::symbolic::symbolic_function;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) async fn execute(operation: LogarithmOperation, value: Value) -> BuiltinResult<Value> {
    if let Some(function) = operation.symbolic_function() {
        if let Some(symbolic) = symbolic_function(&value, function) {
            return Ok(symbolic);
        }
    }
    match value {
        Value::Object(object)
            if operation.accepts_tabular()
                && crate::builtins::table::is_tabular_object(&object) =>
        {
            super::table::evaluate(operation, object).await
        }
        value => execute_non_table(operation, value).await,
    }
}
pub(super) async fn execute_non_table(
    operation: LogarithmOperation,
    value: Value,
) -> BuiltinResult<Value> {
    super::extensions::ensure(operation, &value).await?;
    match value {
        Value::GpuTensor(handle) => super::provider::evaluate(operation, handle).await,
        value => super::host::evaluate(operation, value),
    }
}
