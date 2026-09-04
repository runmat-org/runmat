use runmat_value::Value;

use crate::builtins::table::select_rows;

use super::{GroupColumn, VariableResult};
use crate::builtins::array::grouping::keys::GroupIndex;

pub(crate) fn label_columns(
    columns: &[GroupColumn],
    index: &GroupIndex,
) -> VariableResult<Vec<Value>> {
    let rows = index
        .first_rows
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "an empty group has no source label row".to_string())?;
    columns
        .iter()
        .map(|column| select_group_rows(&column.value, &rows))
        .collect()
}

pub(crate) fn select_group_rows(value: &Value, rows: &[usize]) -> VariableResult<Value> {
    if matches!(value, Value::Object(object) if object.is_class(runmat_types::standard::CALENDAR_DURATION))
    {
        let (months, days) = crate::builtins::datetime::calendar_duration_tensors_from_value(value)
            .map_err(|error| error.to_string())?;
        let months =
            select_rows(&Value::Tensor(months), rows).map_err(|error| error.to_string())?;
        let days = select_rows(&Value::Tensor(days), rows).map_err(|error| error.to_string())?;
        let (Value::Tensor(months), Value::Tensor(days)) = (months, days) else {
            return Err("calendarDuration row selection produced invalid storage".into());
        };
        return crate::builtins::datetime::calendar_duration_object_from_tensors(months, days)
            .map_err(|error| error.to_string());
    }
    select_rows(value, rows).map_err(|error| error.to_string())
}
