use runmat_value::{ObjectInstance, Value};

use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) async fn evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    operation: ExponentialOperation,
    object: ObjectInstance,
) -> BuiltinResult<Value> {
    // A table's numeric columns are nested operations, not confirmation that
    // the enclosing VM instruction executed a closed numeric host branch.
    let nested = runtime.map(|runtime| runtime.clone().without_vm_fusion_host_route());
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|error| super::errors::invalid(operation, error.message))?;
    let mut output = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in variables.into_fields_unbounded() {
        super::extensions::ensure(operation, &value)?;
        if matches!(value, Value::Object(_)) {
            return Err(super::errors::invalid(
                operation,
                format!(
                    "table variable {name} does not support {}",
                    operation.name()
                ),
            ));
        }
        output.insert(
            name,
            super::engine::execute_non_table(nested.as_ref(), operation, value).await?,
        );
    }
    let output = output.finish();
    crate::builtins::table::table_replace_variables_like(&object, output)
        .map_err(|error| super::errors::internal(operation, error.message))
}
