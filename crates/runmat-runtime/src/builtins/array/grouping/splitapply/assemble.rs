use crate::BuiltinResult;

use super::error;

pub(super) async fn outputs(
    groups: Vec<crate::execution::RootedValues>,
) -> BuiltinResult<crate::execution::RootedValues> {
    let mut outputs = crate::execution::RootedValuesBuilder::with_capacity(groups.len());
    for values in groups {
        let output = crate::call_builtin_async("vertcat", values.values())
            .await
            .map_err(|source| {
                error::output(
                    "splitapply: group results must have compatible classes and dimensions",
                    Some(source),
                )
            })?;
        output
            .with_owned(|value| outputs.push(value))
            .map_err(|error| error.into_runtime_error())?;
    }
    outputs.finish().map_err(|error| error.into_runtime_error())
}
