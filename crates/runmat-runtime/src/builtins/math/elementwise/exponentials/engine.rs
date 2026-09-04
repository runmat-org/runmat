use runmat_value::Value;

use crate::builtins::math::symbolic::symbolic_function;
use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) async fn execute(operation: ExponentialOperation, value: Value) -> BuiltinResult<Value> {
    if let Some(function) = operation.symbolic_function() {
        if let Some(symbolic) = symbolic_function(&value, function) {
            return Ok(symbolic);
        }
    }
    super::extensions::ensure(operation, &value)?;
    match value {
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            super::table::evaluate(operation, object).await
        }
        value => execute_non_table(operation, value).await,
    }
}

pub(super) async fn execute_non_table(
    operation: ExponentialOperation,
    value: Value,
) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => super::provider::evaluate(operation, handle).await,
        Value::SparseTensor(sparse) => super::sparse::evaluate(operation, sparse),
        value => super::host::evaluate(operation, value),
    }
}
