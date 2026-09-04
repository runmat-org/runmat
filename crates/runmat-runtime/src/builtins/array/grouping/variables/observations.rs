use runmat_value::{NumericScalar, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::table::value_row_count;

use super::{categorical, temporal, GroupColumn, VariableResult};
use crate::builtins::array::grouping::keys::{GroupIndex, KeyAtom, KeyOrder};

pub(crate) fn build_index(
    columns: &[GroupColumn],
    include_missing: bool,
    order: KeyOrder,
) -> VariableResult<GroupIndex> {
    let rows = columns
        .first()
        .map(|column| column.rows)
        .ok_or_else(|| "expected at least one grouping variable".to_string())?;
    if let Some(column) = columns.iter().find(|column| column.rows != rows) {
        return Err(format!(
            "grouping variables must have matching row counts ({rows} and {})",
            column.rows
        ));
    }
    let rows = (0..rows)
        .map(|row| {
            let key = columns
                .iter()
                .map(|column| column.atom(row).cloned())
                .collect::<VariableResult<Vec<_>>>()?;
            Ok(
                (include_missing || key.iter().all(|atom| atom != &KeyAtom::Missing))
                    .then_some(key),
            )
        })
        .collect::<VariableResult<Vec<_>>>()?;
    GroupIndex::build(rows, order)
}

pub(super) fn validate_and_count(value: &Value) -> VariableResult<usize> {
    match value {
        Value::Tensor(value) => ensure_vector(&value.shape, "numeric grouping variable")?,
        Value::LogicalArray(value) => ensure_vector(&value.shape, "logical grouping variable")?,
        Value::StringArray(value) => ensure_vector(&value.shape, "string grouping variable")?,
        Value::Cell(value) => {
            ensure_vector(&[value.rows, value.cols], "cell grouping variable")?;
            if !value
                .data
                .iter()
                .all(|item| matches!(item, Value::CharArray(chars) if chars.rows <= 1))
            {
                return Err("cell grouping variables must contain character vectors".into());
            }
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => {}
        Value::Object(object)
            if categorical::is_supported_object(object)
                || temporal::is_supported_object(object) => {}
        Value::SparseTensor(_) => return Err("sparse grouping variables are not supported".into()),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err("complex grouping variables are not supported".into())
        }
        other => return Err(format!("unsupported grouping variable {other:?}")),
    }
    row_count(value)
}

fn ensure_vector(shape: &[usize], context: &str) -> VariableResult<()> {
    if shape.iter().filter(|dimension| **dimension > 1).count() <= 1 {
        Ok(())
    } else {
        Err(format!("{context} must be a vector"))
    }
}

fn row_count(value: &Value) -> VariableResult<usize> {
    if matches!(value, Value::Object(object) if object.is_class(runmat_types::standard::CALENDAR_DURATION))
    {
        let (months, days) = crate::builtins::datetime::calendar_duration_tensors_from_value(value)
            .map_err(|error| error.to_string())?;
        if months.shape != days.shape {
            return Err("calendarDuration component shapes do not match".into());
        }
        ensure_vector(&months.shape, "calendarDuration grouping variable")?;
        return Ok(tensor_utils::tensor_element_len(&months));
    }
    value_row_count(value).map_err(|error| error.to_string())
}

pub(super) fn atoms(value: &Value, rows: usize) -> VariableResult<Vec<KeyAtom>> {
    if matches!(value, Value::Object(object) if categorical::is_supported_object(object)) {
        return categorical::atoms(value, rows);
    }
    if matches!(value, Value::Object(object) if temporal::is_supported_object(object)) {
        return temporal::atoms(value, rows);
    }
    (0..rows).map(|row| atom_at(value, row)).collect()
}

fn atom_at(value: &Value, row: usize) -> VariableResult<KeyAtom> {
    match value {
        Value::Tensor(value) => value
            .numeric_value_at(row)
            .map(|value| KeyAtom::from_numeric(value).unwrap_or(KeyAtom::Missing))
            .ok_or_else(|| "numeric grouping row is out of bounds".into()),
        Value::Num(value) if row == 0 => Ok(number(*value)),
        Value::Int(value) if row == 0 => Ok(KeyAtom::Integer(value.clone())),
        Value::LogicalArray(value) => value
            .data
            .get(row)
            .map(|value| KeyAtom::Logical(*value != 0))
            .ok_or_else(|| "logical grouping row is out of bounds".into()),
        Value::Bool(value) if row == 0 => Ok(KeyAtom::Logical(*value)),
        Value::StringArray(value) => value
            .data
            .get(row)
            .map(|value| KeyAtom::from_text(value).unwrap_or(KeyAtom::Missing))
            .ok_or_else(|| "string grouping row is out of bounds".into()),
        Value::String(value) if row == 0 => {
            Ok(KeyAtom::from_text(value).unwrap_or(KeyAtom::Missing))
        }
        Value::Cell(value) => value
            .data
            .get(row)
            .map(cellstr_atom)
            .transpose()?
            .ok_or_else(|| "cell grouping row is out of bounds".into()),
        _ => Err("grouping row is out of bounds or has unsupported storage".into()),
    }
}

fn number(value: f64) -> KeyAtom {
    KeyAtom::from_numeric(NumericScalar::F64(value)).unwrap_or(KeyAtom::Missing)
}

fn cellstr_atom(value: &Value) -> VariableResult<KeyAtom> {
    let Value::CharArray(value) = value else {
        return Err("cell grouping variables must contain character vectors".into());
    };
    let text = value.data.iter().collect::<String>();
    Ok(if text.is_empty() {
        KeyAtom::Missing
    } else {
        KeyAtom::Text(text)
    })
}
