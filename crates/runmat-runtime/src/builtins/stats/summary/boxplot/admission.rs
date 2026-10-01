//! Class and geometry facts admitted before resident boxplot inputs transfer.
use super::*;
use runmat_accelerate_api::GpuTensorStorage;

pub(super) fn admit_host_visible(x: &Value, group: Option<&Value>) -> BuiltinResult<()> {
    let (shape, count) = match x {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => (None, 1),
        Value::Tensor(tensor) => (Some(tensor.shape.as_slice()), tensor.len()),
        Value::LogicalArray(array) => (Some(array.shape.as_slice()), array.len()),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != GpuTensorStorage::Real {
                return Err(invalid("boxplot: X must be real numeric data"));
            }
            (
                Some(handle.shape()),
                handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                    .ok_or_else(|| invalid("boxplot: X shape overflows"))?,
            )
        }
        _ => return Err(invalid("boxplot: X must be numeric data")),
    };
    if count == 0 || shape.is_some_and(|shape| shape.len() > 2) {
        return Err(invalid("boxplot: X must be a nonempty vector or matrix"));
    }
    if let Some(group) = group {
        validate_known_group(group, shape, count)?;
    }
    Ok(())
}

fn validate_known_group(
    group: &Value,
    x_shape: Option<&[usize]>,
    x_count: usize,
) -> BuiltinResult<()> {
    let count = match group {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Some(1),
        Value::Tensor(tensor) => Some(tensor.len()),
        Value::LogicalArray(array) => Some(array.len()),
        Value::StringArray(array) => Some(array.data.len()),
        Value::CharArray(array) => Some(array.rows),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != GpuTensorStorage::Real {
                return Err(invalid(
                    "boxplot: grouping data must be real numeric or text",
                ));
            }
            Some(
                handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                    .ok_or_else(|| invalid("boxplot: group shape overflows"))?,
            )
        }
        Value::Cell(_) => None,
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => None,
        _ => {
            return Err(invalid(
                "boxplot: grouping data must be numeric, text, cell, or categorical",
            ))
        }
    };
    if let Some(group_count) = count {
        let rows = x_shape.and_then(|shape| shape.first().copied());
        let matrix_rows =
            x_shape.is_some_and(|shape| shape.len() == 2 && shape[0] > 1 && shape[1] > 1);
        if group_count != x_count && !(matrix_rows && rows == Some(group_count)) {
            return Err(invalid("boxplot: grouping variable length must match X"));
        }
    }
    Ok(())
}
