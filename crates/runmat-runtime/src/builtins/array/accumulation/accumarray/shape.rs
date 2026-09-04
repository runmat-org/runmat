use runmat_value::Value;

use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

use super::{error, subscripts};

pub(super) const MAX_MATERIALIZED_ELEMENTS: usize = 50_000_000;

pub(super) fn resolve(
    index_rows: &[Vec<usize>],
    explicit: Option<&Value>,
) -> BuiltinResult<Vec<usize>> {
    match explicit.filter(|value| !is_empty(value)) {
        Some(value) => parse_explicit(value),
        None => Ok(infer(index_rows)),
    }
}

fn parse_explicit(value: &Value) -> BuiltinResult<Vec<usize>> {
    let mut dimensions = match value {
        Value::Int(value) => vec![subscripts::positive_exact(value, "accumarray size")?],
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => tensor
            .integer_storage()
            .into_iter()
            .flat_map(|storage| storage.exact_values())
            .map(|value| subscripts::positive_exact(&value, "accumarray size"))
            .collect::<BuiltinResult<Vec<_>>>()?,
        Value::Tensor(tensor) => tensor_utils::tensor_values_f64(tensor)
            .into_iter()
            .map(|value| subscripts::positive_float(value, "accumarray size"))
            .collect::<BuiltinResult<Vec<_>>>()?,
        Value::Num(value) => vec![subscripts::positive_float(*value, "accumarray size")?],
        other => {
            return Err(error::invalid(format!(
                "accumarray size: expected numeric vector, got {other:?}"
            )))
        }
    };
    if dimensions.is_empty() {
        return Err(error::invalid(
            "accumarray size: size vector must not be empty",
        ));
    }
    if dimensions.len() == 1 {
        dimensions.push(1);
    }
    Ok(dimensions)
}

fn infer(index_rows: &[Vec<usize>]) -> Vec<usize> {
    let rank = index_rows.first().map(Vec::len).unwrap_or(1).max(1);
    let mut shape = vec![0; rank];
    for row in index_rows {
        for (dimension, index) in row.iter().enumerate() {
            shape[dimension] = shape[dimension].max(*index);
        }
    }
    if rank == 1 {
        vec![shape[0], 1]
    } else {
        shape
    }
}

pub(super) fn linear_index(subscripts: &[usize], shape: &[usize]) -> BuiltinResult<usize> {
    if subscripts.len() > shape.len() {
        return Err(error::invalid("accumarray: too many subscript dimensions"));
    }
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (dimension, size) in shape.iter().copied().enumerate() {
        let subscript = subscripts.get(dimension).copied().unwrap_or(1);
        if subscript == 0 || subscript > size {
            return Err(error::invalid("accumarray: subscript exceeds output size"));
        }
        linear = linear
            .checked_add(
                (subscript - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| error::too_large("accumarray: output linear index overflow"))?,
            )
            .ok_or_else(|| error::too_large("accumarray: output linear index overflow"))?;
        stride = stride
            .checked_mul(size)
            .ok_or_else(|| error::too_large("accumarray: output size overflow"))?;
    }
    Ok(linear)
}

pub(super) fn element_count(shape: &[usize]) -> BuiltinResult<usize> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| error::too_large("accumarray: output size overflow"))?;
    if count > MAX_MATERIALIZED_ELEMENTS {
        Err(error::too_large("accumarray: output is too large"))
    } else {
        Ok(count)
    }
}

pub(super) fn rows_and_columns(shape: &[usize]) -> (usize, usize) {
    let rows = shape.first().copied().unwrap_or(0);
    let columns = shape
        .get(1..)
        .map(|dimensions| dimensions.iter().product())
        .unwrap_or(1);
    (rows, columns)
}

pub(super) fn is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor) == 0,
        Value::StringArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::CharArray(chars) => chars.data.is_empty(),
        _ => false,
    }
}

pub(super) fn binary_flag(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(value) if *value == 0.0 => Ok(false),
        Value::Num(value) if *value == 1.0 => Ok(true),
        Value::Int(value) if value.is_zero() => Ok(false),
        Value::Int(value) if value.try_to_usize() == Some(1) => Ok(true),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(error::invalid(format!(
            "accumarray issparse: expected logical or numeric 0 or 1, got {other:?}"
        ))),
    }
}
