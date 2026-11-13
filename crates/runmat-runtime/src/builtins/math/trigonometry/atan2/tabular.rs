//! Own numeric column results until the complete tabular value is published.
use super::*;
use crate::execution::{RootedValues, RootedValuesBuilder};
use runmat_value::ObjectInstance;

pub(super) async fn execute(
    source: ObjectInstance,
    variables: Vec<(String, Value)>,
    right_values: RootedValues,
) -> BuiltinResult<RootedValue> {
    for ((_, y), x) in variables.iter().zip(right_values.values()) {
        admission::pair(y, x)?;
    }
    let borrowed: Vec<_> = variables
        .iter()
        .map(|(_, y)| y)
        .chain(right_values.values().iter())
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle.clone())
            } else {
                None
            }
        })
        .collect();
    right_values
        .with_owned_output_storage_async(|rights| async move {
            let rights = rights.try_into_owned().map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
            let mut protected = borrowed;
            let mut names = Vec::with_capacity(variables.len());
            let mut columns = RootedValuesBuilder::with_capacity(variables.len());
            for ((name, y), x) in variables.into_iter().zip(rights) {
                let output = atan2_admitted(y, x, &protected).await?;
                if let Value::GpuTensor(handle) = &output {
                    protected.push(handle.clone());
                }
                names.push(name);
                let owned = RootedValue::new(output).map_err(|cause| cause.into_runtime_error())?;
                columns.push_owned(owned);
            }
            let columns = columns
                .finish()
                .map_err(|cause| cause.into_runtime_error())?;
            crate::builtins::table::finish_binary_owned(&source, names, columns)
                .map_err(|error| atan2_error_with_detail(&ATAN2_ERROR_INVALID_INPUT, error))
        })
        .await
}

#[cfg(test)]
#[path = "tabular_tests.rs"]
mod tests;
