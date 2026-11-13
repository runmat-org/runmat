use runmat_value::Value;

use crate::builtins::array::grouping::variables::build_index;
use crate::builtins::common::grouping_keys::KeyOrder;
use crate::{gather_if_needed_async, BuiltinResult};

use super::{error, extensions, input::FindGroupsInput, output};

pub(super) async fn apply(
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    extensions::validate(&first, rest.view())?;
    let mut values = Vec::with_capacity(rest.len() + 1);
    values.push(gather_if_needed_async(&first).await?);
    for value in rest {
        values.push(gather_if_needed_async(&value).await?);
    }
    let input = FindGroupsInput::prepare(values)?;
    let index = build_index(input.columns(), false, KeyOrder::Sorted)
        .map_err(|source| source.attach(error::invalid("invalid grouping variable")))?;
    output::build(&input, &index)
}
