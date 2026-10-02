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

/// An ungrouped vector or matrix still contributes one synthetic factor
/// (its vector label or matrix column). A non-cell group also contributes
/// one factor; cell inputs can encode either
/// per-observation scalar labels or multiple factor vectors, so their count
/// remains unknown until the established group decoder resolves that role.
pub(super) fn known_factor_count(group: Option<&Value>) -> Option<usize> {
    match group {
        None => Some(1),
        Some(Value::Cell(_)) => None,
        Some(_) => Some(1),
    }
}

fn validate_known_group(
    group: &Value,
    x_shape: Option<&[usize]>,
    x_count: usize,
) -> BuiltinResult<()> {
    validate_group_leaf_classes(group)?;
    if matches!(group, Value::Cell(_) | Value::Object(_)) && !value_contains_gpu(group)? {
        return super::groups::validate_known_host_group(group, x_shape, x_count);
    }
    if matches!(group, Value::Cell(_)) {
        let matches_elements = could_match_group_length(group, x_count)?;
        let matches_rows = if matches_elements {
            false
        } else if let Some(shape) =
            x_shape.filter(|shape| shape.len() == 2 && shape[0] > 1 && shape[1] > 1)
        {
            could_match_group_length(group, shape[0])?
        } else {
            false
        };
        if !matches_elements && !matches_rows {
            return Err(invalid("boxplot: grouping variable length must match X"));
        }
    }
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

/// A resident leaf keeps its known class and shape through materialization.
/// A cell is either one scalar label per row or a list of grouping vectors;
/// reject only when neither role can match the sample geometry.
fn could_match_group_length(value: &Value, expected: usize) -> BuiltinResult<bool> {
    match value {
        Value::Cell(cell) => {
            if cell.data.len() == expected && cell.data.iter().all(could_be_scalar_label) {
                return Ok(true);
            }
            for element in &cell.data {
                if !could_match_group_length(element, expected)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Ok(expected == 1),
        Value::Tensor(tensor) => Ok(tensor.len() == expected),
        Value::LogicalArray(array) => Ok(array.len() == expected),
        Value::StringArray(array) => Ok(array.data.len() == expected),
        Value::CharArray(array) => Ok(array.rows == expected),
        Value::GpuTensor(handle) => Ok(handle
            .shape()
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            == Some(expected)),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            // Host categorical row count belongs to the table domain. A
            // resident property may become valid after gather, so keep that
            // case dynamic rather than interpreting object field strings here.
            if value_contains_gpu(value)? {
                return Ok(true);
            }
            let labels =
                crate::builtins::table::categorical_observation_labels(object).map_err(|err| {
                    invalid_with_source("boxplot: invalid categorical grouping variable", err)
                })?;
            Ok(labels.len() == expected)
        }
        _ => Ok(false),
    }
}

fn could_be_scalar_label(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => true,
        Value::Tensor(tensor) => tensor.len() == 1,
        Value::LogicalArray(array) => array.len() == 1,
        Value::StringArray(array) => array.data.len() == 1,
        Value::CharArray(array) => array.rows == 1,
        Value::GpuTensor(handle) => {
            handle
                .shape()
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                == Some(1)
        }
        _ => false,
    }
}

fn validate_group_leaf_classes(group: &Value) -> BuiltinResult<()> {
    match group {
        Value::Cell(cell) => {
            for value in &cell.data {
                validate_group_leaf_classes(value)?;
            }
            Ok(())
        }
        Value::GpuTensor(handle) if handle.physical().storage() != GpuTensorStorage::Real => Err(
            invalid("boxplot: grouping data must be real numeric or text"),
        ),
        Value::Object(object) if !object.is_class(runmat_types::standard::CATEGORICAL) => {
            Err(invalid("boxplot: grouping data must be categorical"))
        }
        Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::SparseTensor(_)
        | Value::Struct(_) => Err(invalid(
            "boxplot: grouping data must be real numeric or text",
        )),
        _ => Ok(()),
    }
}
