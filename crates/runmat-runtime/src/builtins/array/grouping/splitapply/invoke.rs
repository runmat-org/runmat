use runmat_value::Value;

use crate::{call_feval_async_with_outputs, gather_if_needed_async, BuiltinResult};

use super::error;
use super::groups::Groups;

pub(super) async fn groups(
    function: Value,
    data: &[Value],
    groups: &Groups,
    requested_outputs: usize,
) -> BuiltinResult<Vec<crate::execution::RootedValues>> {
    let _inputs_root = crate::execution::RootedValueLease::for_values(
        std::iter::once(&function).chain(data.iter()),
    )
    .map_err(|error| error.into_runtime_error())?;
    let mut collectors = (0..requested_outputs)
        .map(|_| crate::execution::RootedValuesBuilder::with_capacity(groups.rows_by_group.len()))
        .collect::<Vec<_>>();
    for indices in &groups.rows_by_group {
        let arguments = data
            .iter()
            .map(|value| super::slice::select(value, groups, indices))
            .collect::<BuiltinResult<Vec<_>>>()?;
        let result = call_feval_async_with_outputs(function.clone(), &arguments, requested_outputs)
            .await
            .map_err(|source| error::callback("splitapply: group function failed", Some(source)))?;
        let outputs = result
            .with_owned_async(|result| async move {
                let outputs = normalize(result, requested_outputs)?;
                let outputs = outputs.try_into_owned().map_err(|_| {
                    error::callback(
                        "splitapply: shared callback outputs require separate admission",
                        None,
                    )
                })?;
                let mut rooted = Vec::with_capacity(outputs.as_slice().len());
                for output in outputs {
                    rooted.push(
                        crate::execution::RootedValue::new(gather_if_needed_async(&output).await?)
                            .map_err(|error| error.into_runtime_error())?,
                    );
                }
                Ok::<_, crate::RuntimeError>(rooted)
            })
            .await?;
        for (collector, output) in collectors.iter_mut().zip(outputs) {
            output
                .with_owned(|value| collector.push(value))
                .map_err(|error| error.into_runtime_error())?;
        }
    }
    collectors
        .into_iter()
        .map(|collector| {
            collector
                .finish()
                .map_err(|error| error.into_runtime_error())
        })
        .collect()
}

fn normalize(value: Value, requested: usize) -> BuiltinResult<runmat_value::OutputListStorage> {
    match value {
        Value::OutputList(values) if values.len() == requested => Ok(values),
        Value::OutputList(values) => Err(error::callback(
            format!(
                "splitapply: group function returned {} outputs; {requested} were requested",
                values.len()
            ),
            None,
        )),
        value if requested == 1 => Ok(runmat_value::OutputListStorage::unbounded(vec![value])),
        _ => Err(error::callback(
            "splitapply: group function did not return the requested outputs",
            None,
        )),
    }
}
