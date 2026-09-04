use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::{empty_groups, extensions, input::GroupCountInput, output};

pub(super) async fn apply(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    extensions::validate(&first, &rest)?;
    let first = gather_if_needed_async(&first).await?;
    let mut host_rest = Vec::with_capacity(rest.len());
    for value in rest {
        host_rest.push(gather_if_needed_async(&value).await?);
    }
    let input = GroupCountInput::prepare(first, host_rest)?;
    let index = empty_groups::build(
        &input.columns,
        input.options.include_missing,
        input.options.include_empty,
        input.bins.as_ref(),
    )?;
    output::build(&input, &index)
}
