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
    match value {
        Value::GpuTensor(handle) => super::provider::evaluate(operation, handle).await,
        Value::SparseTensor(sparse) => super::sparse::evaluate(operation, sparse),
        value => {
            let route = crate::builtins::math::host_route::canonical(
                runtime,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
                std::iter::once(&value),
            );
            let output = super::host::evaluate(operation, value)?;
            if let Some(route) = route {
                route.confirm_host_branch(
                    runtime.unwrap(),
                    crate::context::VmFusionHostInstruction::ElementwiseCall,
                );
            }
            Ok(output)
        }
    }
}
