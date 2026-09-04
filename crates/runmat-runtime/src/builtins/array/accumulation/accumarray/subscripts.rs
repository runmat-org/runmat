use runmat_value::{IntValue, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

use super::error;

pub(super) fn parse(value: Value) -> BuiltinResult<Vec<Vec<usize>>> {
    match value {
        Value::Int(value) => Ok(vec![vec![positive_exact(&value, "accumarray subscript")?]]),
        Value::Num(value) => Ok(vec![vec![positive_float(value, "accumarray subscript")?]]),
        Value::Tensor(tensor) if tensor_utils::tensor_element_len(&tensor) == 0 => Ok(Vec::new()),
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            if let Some(storage) = tensor.integer_storage() {
                if cols <= 1 || rows == 1 {
                    return storage
                        .exact_values()
                        .iter()
                        .map(|value| Ok(vec![positive_exact(value, "accumarray subscript")?]))
                        .collect();
                }
                return (0..rows)
                    .map(|row| {
                        (0..cols)
                            .map(|col| {
                                storage
                                    .value_at(row + col * rows)
                                    .ok_or_else(|| {
                                        error::invalid(
                                            "accumarray: integer subscript index out of bounds",
                                        )
                                    })
                                    .and_then(|value| {
                                        positive_exact(&value, "accumarray subscript")
                                    })
                            })
                            .collect()
                    })
                    .collect();
            }
            if cols <= 1 || rows == 1 {
                return tensor_utils::tensor_into_values_f64(tensor)
                    .into_iter()
                    .map(|value| Ok(vec![positive_float(value, "accumarray subscript")?]))
                    .collect();
            }
            (0..rows)
                .map(|row| {
                    (0..cols)
                        .map(|col| {
                            tensor
                                .get2(row, col)
                                .map_err(error::invalid)
                                .and_then(|value| positive_float(value, "accumarray subscript"))
                        })
                        .collect()
                })
                .collect()
        }
        Value::Cell(cell) => parse_cell_columns(cell.data),
        other => Err(error::invalid(format!(
            "accumarray: unsupported subscript input {other:?}"
        ))),
    }
}

fn parse_cell_columns(values: Vec<Value>) -> BuiltinResult<Vec<Vec<usize>>> {
    let mut columns = Vec::with_capacity(values.len());
    for value in values {
        let column = parse(value)?;
        if column.iter().any(|row| row.len() != 1) {
            return Err(error::invalid(
                "accumarray: cell subscript entries must be vectors",
            ));
        }
        columns.push(column.into_iter().map(|row| row[0]).collect::<Vec<_>>());
    }
    let rows = columns.first().map(Vec::len).unwrap_or(0);
    if columns.iter().any(|column| column.len() != rows) {
        return Err(error::invalid(
            "accumarray: cell subscript vectors must have equal length",
        ));
    }
    Ok((0..rows)
        .map(|row| columns.iter().map(|column| column[row]).collect())
        .collect())
}

pub(super) fn positive_float(value: f64, context: &str) -> BuiltinResult<usize> {
    platform_usize(value)
        .filter(|value| *value > 0)
        .ok_or_else(|| error::invalid(format!("{context}: expected positive integer")))
}

pub(super) fn positive_exact(value: &IntValue, context: &str) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|value| *value > 0)
        .ok_or_else(|| error::invalid(format!("{context}: expected positive integer")))
}

fn platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        None
    } else {
        Some(value as usize)
    }
}
