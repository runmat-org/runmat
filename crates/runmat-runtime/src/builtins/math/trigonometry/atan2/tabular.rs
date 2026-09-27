//! Own numeric column results until the complete tabular value is published.
use super::*;
use crate::builtins::common::resident_output::ResidentResultGuard;
use runmat_value::ObjectInstance;

pub(super) async fn execute(
    source: ObjectInstance,
    variables: Vec<(String, Value, Value)>,
) -> BuiltinResult<Value> {
    for (_, y, x) in &variables {
        admission::pair(y, x)?;
    }
    let borrowed: Vec<_> = variables
        .iter()
        .flat_map(|(_, y, x)| [y, x])
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle.clone())
            } else {
                None
            }
        })
        .collect();
    let borrowed_refs: Vec<_> = borrowed.iter().collect();
    let mut protected = borrowed.clone();
    let mut owned = Vec::new();
    let mut columns = Vec::with_capacity(variables.len());
    for (name, y, x) in variables {
        let output = atan2_admitted(y, x, &protected).await?;
        if let Value::GpuTensor(handle) = &output {
            let guard = ResidentResultGuard::new(&output, &borrowed_refs)
                .map_err(|error| atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, error))?;
            protected.push(handle.clone());
            owned.push((guard, Value::GpuTensor(handle.clone())));
        }
        columns.push((name, output));
    }
    let result = crate::builtins::table::finish_binary(&source, columns)
        .map_err(|error| atan2_error_with_detail(&ATAN2_ERROR_INVALID_INPUT, error))?;
    for (guard, column) in owned {
        guard.publish(&column);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "tabular_tests.rs"]
mod tests;
