use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::input::Input;

pub(super) async fn apply(
    function: Value,
    first_data: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let _inputs_root = crate::execution::RootedValueLease::for_values(
        [&function, &first_data].into_iter().chain(rest.iter()),
    )
    .map_err(|error| error.into_runtime_error())?;
    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    let input = Input::prepare(
        gather_if_needed_async(&function).await?,
        gather_if_needed_async(&first_data).await?,
        gathered,
    )?;
    let requested = crate::output_count::current_output_count()
        .unwrap_or(1)
        .max(1);
    let collected =
        super::invoke::groups(input.function, &input.data, &input.groups, requested).await?;
    let outputs = super::assemble::outputs(collected).await?;
    outputs.with_unbounded_values(|outputs| {
        let value = super::super::requested_outputs::finish(outputs)?;
        crate::execution::RootedValue::new(value).map_err(|error| error.into_runtime_error())
    })?
}
