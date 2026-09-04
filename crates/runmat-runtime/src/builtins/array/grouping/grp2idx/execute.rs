use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::{input::GroupingInput, output, provider};

pub(super) async fn apply(value: Value) -> BuiltinResult<Value> {
    let resident = provider::resident_input(&value);
    let host = gather_if_needed_async(&value).await?;
    let input = GroupingInput::prepare(host)?;
    let index = input.index()?;
    let g = output::indices(&index)?;
    let gn = output::names(&index)?;
    let gl = input.levels(&index)?;
    let (g, gl) = provider::restore(resident.as_ref(), g, gl)?;
    Ok(Value::OutputList(vec![g, gn, gl]))
}
