use runmat_value::Value;

use crate::{call_feval_async_with_outputs, gather_if_needed_async, BuiltinResult};

use super::error;
use super::groups::Groups;

pub(super) async fn groups(
    function: Value,
    data: &[Value],
    groups: &Groups,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Vec<Value>>> {
    let mut collectors = (0..requested_outputs)
        .map(|_| Vec::with_capacity(groups.rows_by_group.len()))
        .collect::<Vec<_>>();
    for indices in &groups.rows_by_group {
        let arguments = data
            .iter()
            .map(|value| super::slice::select(value, groups, indices))
            .collect::<BuiltinResult<Vec<_>>>()?;
        let result = call_feval_async_with_outputs(function.clone(), &arguments, requested_outputs)
            .await
            .map_err(|source| error::callback("splitapply: group function failed", Some(source)))?;
        let outputs = normalize(result, requested_outputs)?;
        for (collector, output) in collectors.iter_mut().zip(outputs) {
            collector.push(gather_if_needed_async(&output).await?);
        }
    }
    Ok(collectors)
}

fn normalize(value: Value, requested: usize) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::OutputList(values) if values.len() == requested => Ok(values),
        Value::OutputList(values) => Err(error::callback(
            format!(
                "splitapply: group function returned {} outputs; {requested} were requested",
                values.len()
            ),
            None,
        )),
        value if requested == 1 => Ok(vec![value]),
        _ => Err(error::callback(
            "splitapply: group function did not return the requested outputs",
            None,
        )),
    }
}
