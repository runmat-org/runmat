use runmat_value::{NumericScalar, ObjectInstance, Tensor, Value};

use super::VariableResult;
use crate::builtins::array::grouping::keys::KeyAtom;
use crate::builtins::common::tensor::tensor_element_len;

pub(super) fn is_supported_object(value: &ObjectInstance) -> bool {
    value.is_class(runmat_types::standard::DATETIME)
        || value.is_class(runmat_types::standard::DURATION)
        || value.is_class(runmat_types::standard::CALENDAR_DURATION)
}

pub(super) fn atoms(value: &Value, rows: usize) -> VariableResult<Vec<KeyAtom>> {
    let Value::Object(object) = value else {
        return Err("temporal grouping storage is not an object".into());
    };
    if object.is_class(runmat_types::standard::DATETIME) {
        let values = crate::builtins::datetime::serials_from_datetime_value(value)
            .map_err(|error| error.to_string())?;
        return numeric_atoms(&values, rows, "datetime");
    }
    if object.is_class(runmat_types::standard::DURATION) {
        let values = crate::builtins::duration::duration_tensor_from_duration_value(value)
            .map_err(|error| error.to_string())?;
        return numeric_atoms(&values, rows, "duration");
    }
    if object.is_class(runmat_types::standard::CALENDAR_DURATION) {
        let (months, days) = crate::builtins::datetime::calendar_duration_tensors_from_value(value)
            .map_err(|error| error.to_string())?;
        if tensor_element_len(&months) != rows || tensor_element_len(&days) != rows {
            return Err("calendarDuration grouping storage has an inconsistent shape".into());
        }
        return (0..rows)
            .map(|row| {
                let months = numeric_value_at(&months, row, "calendarDuration months")?;
                let days = numeric_value_at(&days, row, "calendarDuration days")?;
                Ok(if months.is_nan() || days.is_nan() {
                    KeyAtom::Missing
                } else {
                    KeyAtom::CalendarDuration(months, days)
                })
            })
            .collect();
    }
    Err("unsupported temporal grouping object".into())
}

fn numeric_atoms(value: &Tensor, rows: usize, context: &str) -> VariableResult<Vec<KeyAtom>> {
    let actual = tensor_element_len(value);
    if actual != rows {
        return Err(format!(
            "{context} grouping storage has {} observations; expected {rows}",
            actual
        ));
    }
    (0..rows)
        .map(|row| {
            value
                .numeric_value_at(row)
                .map(|value| KeyAtom::from_numeric(value).unwrap_or(KeyAtom::Missing))
                .ok_or_else(|| format!("{context} grouping row is out of bounds"))
        })
        .collect()
}

fn numeric_value_at(value: &Tensor, row: usize, context: &str) -> VariableResult<f64> {
    value
        .numeric_value_at(row)
        .map(NumericScalar::materialize_f64)
        .ok_or_else(|| format!("{context} grouping row is out of bounds"))
}
