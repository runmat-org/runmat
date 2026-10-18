use runmat_value::Value;

use crate::builtins::math::symbolic::symbolic_function;
use crate::BuiltinResult;

use super::operation::ExponentialOperation;

#[cfg(test)]
mod tests;

pub(super) async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    operation: ExponentialOperation,
    value: Value,
) -> BuiltinResult<Value> {
    if let Some(function) = operation.symbolic_function() {
        if let Some(symbolic) = symbolic_function(&value, function) {
            return Ok(symbolic);
        }
    }
    super::extensions::ensure(operation, &value)?;
    match value {
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            super::table::evaluate(runtime, operation, object).await
        }
        value => execute_non_table(runtime, operation, value).await,
    }
}

pub(super) async fn execute_non_table(
    runtime: Option<&crate::context::RuntimeContext>,
    operation: ExponentialOperation,
    value: Value,
) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(super::errors::invalid(
            operation,
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    let output = crate::builtins::math::trigonometry::inverse_helpers::preparation::unary_real(
        runtime,
        operation.catalog_entry().identity,
        value,
        |value: f32| Ok(operation.apply_f32(value)),
        |value: f64| Ok(operation.apply_f64(value)),
        |value| async move {
            let value = match value {
                Value::GpuTensor(handle) => super::provider::evaluate(operation, handle).await,
                Value::SparseTensor(sparse) => super::sparse::evaluate(operation, sparse),
                value => super::host::evaluate(operation, value),
            }?;
            crate::execution::RootedValue::new(value).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await?;
    let (value, _root) = output.into_value_and_lease();
    Ok(value)
}
