use runmat_value::{IntValue, Value};

use super::PartitionPlan;
use crate::builtins::cells::core::block_layout::{checked_element_count, prefix_offsets};
use crate::builtins::cells::core::mat2cell::error::{
    mat2cell_error_with_message, MAT2CELL_ERROR_INVALID_PARTITION, MAT2CELL_ERROR_SIZE_EXCEEDED,
};
use crate::BuiltinResult;

enum Entry {
    Float(f64),
    Integer(IntValue),
}

pub(in crate::builtins::cells::core::mat2cell) fn for_dimensions(
    dims: &[usize],
    arguments: &[Value],
) -> BuiltinResult<PartitionPlan> {
    let mut extents = dims.to_vec();
    extents.resize(extents.len().max(arguments.len()), 1);
    let sizes = extents
        .iter()
        .enumerate()
        .map(|(index, extent)| {
            arguments.get(index).map_or_else(
                || Ok(vec![*extent]),
                |value| vector(value, *extent, index + 1),
            )
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let offsets = sizes
        .iter()
        .map(|values| prefix_offsets(values).ok_or_else(size_exceeded))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut cell_shape = sizes.iter().map(Vec::len).collect::<Vec<_>>();
    match cell_shape.len() {
        0 => cell_shape = vec![1, 1],
        1 => cell_shape.push(1),
        _ => {}
    }
    let cell_count = checked_element_count(&cell_shape).ok_or_else(size_exceeded)?;
    Ok(PartitionPlan {
        sizes,
        offsets,
        cell_shape,
        cell_count,
    })
}

fn vector(value: &Value, extent: usize, dimension: usize) -> BuiltinResult<Vec<usize>> {
    let entries = entries(value)
        .ok_or_else(|| invalid(format!("partition {dimension} must be a numeric vector")))?;
    if entries.is_empty() {
        return if extent == 0 {
            Ok(Vec::new())
        } else {
            Err(invalid(format!(
                "partition sizes for dimension {dimension} must sum to {extent}"
            )))
        };
    }
    let mut sum = 0usize;
    let mut result = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let size = to_usize(entry, dimension, index + 1)?;
        sum = sum.checked_add(size).ok_or_else(size_exceeded)?;
        result.push(size);
    }
    if sum != extent {
        return Err(invalid(format!(
            "partition sizes for dimension {dimension} must sum to {extent} (got {sum})"
        )));
    }
    Ok(result)
}

fn entries(value: &Value) -> Option<Vec<Entry>> {
    match value {
        Value::Num(value) => Some(vec![Entry::Float(*value)]),
        Value::Int(value) => Some(vec![Entry::Integer(value.clone())]),
        Value::Bool(value) => Some(vec![Entry::Float(f64::from(*value))]),
        Value::Tensor(tensor) if vector_shape(&tensor.shape) => {
            tensor.integer_storage().map_or_else(
                || {
                    Some(
                        (0..tensor.len())
                            .map(|index| {
                                Entry::Float(crate::builtins::common::tensor::tensor_value_f64(
                                    tensor, index,
                                ))
                            })
                            .collect(),
                    )
                },
                |storage| {
                    Some(
                        storage
                            .exact_values()
                            .into_iter()
                            .map(Entry::Integer)
                            .collect(),
                    )
                },
            )
        }
        Value::LogicalArray(array) if vector_shape(&array.shape) => Some(
            array
                .data
                .iter()
                .map(|value| Entry::Float(f64::from(*value != 0)))
                .collect(),
        ),
        _ => None,
    }
}

fn to_usize(entry: &Entry, dimension: usize, index: usize) -> BuiltinResult<usize> {
    let error = |requirement: &str| {
        invalid(format!(
            "partition entries must be {requirement} (dimension {dimension}, index {index})"
        ))
    };
    match entry {
        Entry::Integer(value) => value
            .try_to_usize()
            .ok_or_else(|| error("non-negative platform integers")),
        Entry::Float(value) if !value.is_finite() => Err(error("finite")),
        Entry::Float(value) if value.fract() != 0.0 => Err(error("integers")),
        Entry::Float(value) if *value < 0.0 => Err(error("non-negative")),
        Entry::Float(value)
            if *value > usize::MAX as f64 || (usize::BITS == 64 && *value == usize::MAX as f64) =>
        {
            Err(error("within platform limits"))
        }
        Entry::Float(value) => Ok(*value as usize),
    }
}

fn vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|extent| **extent > 1).count() <= 1
}

fn invalid(detail: String) -> crate::RuntimeError {
    mat2cell_error_with_message(
        format!("mat2cell: {detail}"),
        &MAT2CELL_ERROR_INVALID_PARTITION,
    )
}

fn size_exceeded() -> crate::RuntimeError {
    mat2cell_error_with_message(
        "mat2cell: partition size exceeds platform limits",
        &MAT2CELL_ERROR_SIZE_EXCEEDED,
    )
}
