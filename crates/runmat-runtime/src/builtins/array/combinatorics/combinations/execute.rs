use runmat_value::Value;

use crate::builtins::table::table_from_columns;
use crate::BuiltinResult;

use super::{arguments, columns::CombinationColumn, error, plan};

pub(super) async fn apply(
    runtime: Option<&crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let inputs = arguments::gather(runtime, first, rest).await?;
    let _input_owner = inputs.fork();
    let columns = inputs
        .into_iter()
        .map(CombinationColumn::from_value)
        .collect::<BuiltinResult<Vec<_>>>()?;
    let plan = plan::build(
        &columns
            .iter()
            .map(CombinationColumn::len)
            .collect::<Vec<_>>(),
    )?;
    let names = (1..=columns.len())
        .map(|index| format!("Var{index}"))
        .collect();
    let values = columns
        .into_iter()
        .zip(plan.repetitions)
        .map(|(column, repetition)| column.materialize(plan.rows, repetition))
        .collect::<BuiltinResult<Vec<_>>>()?;
    table_from_columns(names, values).map_err(error::internal)
}
