use super::operation::LogarithmOperation;
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use crate::BuiltinResult;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn evaluate(
    runtime: Option<&RuntimeContext>,
    operation: LogarithmOperation,
    object: ObjectInstance,
) -> BuiltinResult<RootedValue> {
    // Per-column math cannot certify the enclosing object/table instruction.
    let nested = runtime.map(|runtime| runtime.clone().without_vm_fusion_host_route());
    let source = Value::Object(object.clone());
    let _source = RootedValueLease::for_values(std::iter::once(&source))
        .map_err(|cause| cause.into_runtime_error())?;
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|error| super::errors::invalid(operation, error.message()))?;
    let mut names = Vec::with_capacity(variables.fields.len());
    let mut output = RootedValuesBuilder::with_capacity(variables.fields.len());
    for (name, value) in variables.into_fields_unbounded() {
        let transformed = super::engine::execute_non_table(nested.as_ref(), operation, value)
            .await
            .map_err(|error| {
                super::errors::invalid(
                    operation,
                    format!(
                        "table variable {name} does not support {}: {}",
                        operation.name(),
                        error.message()
                    ),
                )
            })?;
        names.push(name);
        output.push_owned(transformed);
    }
    let output = output
        .finish()
        .map_err(|cause| cause.into_runtime_error())?;
    crate::builtins::table::table_replace_variables_like_owned(&object, names, output)
        .map_err(|error| super::errors::internal(operation, error.message()))
}
