//! Class and geometry facts admitted before resident boxplot inputs transfer.
use super::*;
use runmat_accelerate_api::GpuTensorStorage;

/// Boxplot's dense matrix projection treats rank zero as 0-by-0 and rank one
/// as a single row, unlike the general Tensor row/column getters.
pub(super) fn boxplot_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) async fn admit_host_visible(
    x: &Value,
    group: Option<&Value>,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<()> {
    let (shape, count) = match x {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => (None, 1),
        Value::Tensor(tensor) => (Some(tensor.shape()), tensor.len()),
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
        validate_known_group(group, shape, count, cooperation).await?;
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

async fn validate_known_group(
    group: &Value,
    x_shape: Option<&[usize]>,
    x_count: usize,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<()> {
    validate_group_leaf_classes(group, cooperation).await?;
    if matches!(group, Value::Cell(_) | Value::Object(_)) && !value_contains_gpu(group)? {
        return super::groups::validate_known_host_group_cooperative(
            group,
            x_shape,
            x_count,
            cooperation,
        )
        .await;
    }
    if matches!(group, Value::Cell(_)) {
        let matches_elements = could_match_group_length(group, x_count, cooperation).await?;
        let matches_rows = if matches_elements {
            false
        } else if let Some(shape) =
            x_shape.filter(|shape| shape.len() == 2 && shape[0] > 1 && shape[1] > 1)
        {
            could_match_group_length(group, shape[0], cooperation).await?
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
        Value::StringArray(array) => Some(array.data().len()),
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
async fn could_match_group_length(
    value: &Value,
    expected: usize,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<bool> {
    let mut pending = vec![std::slice::from_ref(value).iter()];
    while let Some(frame) = pending.last_mut() {
        let Some(value) = frame.next() else {
            pending.pop();
            continue;
        };
        cooperation.completed(1).await?;
        match value {
            Value::Cell(cell) => {
                let mut scalar_labels = cell.data().len() == expected;
                if scalar_labels {
                    for item in cell.data() {
                        scalar_labels &= could_be_scalar_label(item);
                        cooperation.completed(1).await?;
                        if !scalar_labels {
                            break;
                        }
                    }
                }
                if !scalar_labels {
                    pending.try_reserve(1).map_err(|error| {
                        allocation_failure(
                            "boxplot: grouping metadata exceeds available memory",
                            error,
                        )
                    })?;
                    pending.push(cell.data().iter());
                }
            }
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => {
                if expected != 1 {
                    return Ok(false);
                }
            }
            Value::Tensor(tensor) if tensor.len() == expected => {}
            Value::LogicalArray(array) if array.len() == expected => {}
            Value::StringArray(array) if array.data().len() == expected => {}
            Value::CharArray(array) if array.rows == expected => {}
            Value::GpuTensor(handle)
                if handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                    == Some(expected) => {}
            Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
                // Resident categorical properties remain deferred to the owner.
                if value_contains_gpu(value)? {
                    continue;
                }
                let view =
                    crate::builtins::table::categorical_label_view(object).map_err(|err| {
                        invalid_with_source("boxplot: invalid categorical grouping variable", err)
                    })?;
                for index in 0..view.len() {
                    view.observation_label_at(index).map_err(|err| {
                        invalid_with_source("boxplot: invalid categorical grouping variable", err)
                    })?;
                    cooperation.completed(1).await?;
                }
                if view.len() != expected {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
}

fn could_be_scalar_label(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => true,
        Value::Tensor(tensor) => tensor.len() == 1,
        Value::LogicalArray(array) => array.len() == 1,
        Value::StringArray(array) => array.data().len() == 1,
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

async fn validate_group_leaf_classes(
    group: &Value,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<()> {
    let mut pending = vec![std::slice::from_ref(group).iter()];
    while let Some(frame) = pending.last_mut() {
        let Some(value) = frame.next() else {
            pending.pop();
            continue;
        };
        cooperation.completed(1).await?;
        match value {
            Value::Cell(cell) => {
                pending.try_reserve(1).map_err(|error| {
                    allocation_failure("boxplot: grouping metadata exceeds available memory", error)
                })?;
                pending.push(cell.data().iter());
            }
            Value::GpuTensor(handle) if handle.physical().storage() != GpuTensorStorage::Real => {
                return Err(invalid(
                    "boxplot: grouping data must be real numeric or text",
                ));
            }
            Value::Object(object) if !object.is_class(runmat_types::standard::CATEGORICAL) => {
                return Err(invalid("boxplot: grouping data must be categorical"));
            }
            Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::SparseTensor(_)
            | Value::Struct(_) => {
                return Err(invalid(
                    "boxplot: grouping data must be real numeric or text",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}
