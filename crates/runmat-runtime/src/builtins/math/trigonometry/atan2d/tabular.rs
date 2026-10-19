//! Keep column values rooted until the complete degree-valued table is published.
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
    for ((_, y), x) in variables.iter().zip(right_values.values()) {
        admission::host_integer_exactness(y, x)?;
    }
    for ((_, y), x) in variables.iter().zip(right_values.values()) {
        admission::exact_integer_boundaries(y, x).await?;
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
                let output = atan2d_admitted(y, x, &protected).await?;
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
                .map_err(|cause| error(&ATAN2D_ERROR_INVALID_INPUT, cause))
        })
        .await
}
