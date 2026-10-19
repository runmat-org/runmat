//! Scatter object assignments into a grown column-major shape.
//!
//! Newly addressed slots must be supplied by the assignment. Missing objects
//! require class construction, which this storage operation does not perform.

use runmat_value::{ObjectArray, Value};

use crate::indexing::{plan::IndexPlan, shape::column_major_strides};
use crate::{runtime_error::semantic_error, RuntimeError};

pub(super) fn assign(
    array: &ObjectArray,
    plan: &IndexPlan,
    replacements: Vec<Value>,
) -> Result<Vec<Value>, RuntimeError> {
    if array.shape() == plan.base_shape {
        let mut data = array.data().to_vec();
        for (index, value) in plan.indices.iter().zip(replacements) {
            *data.get_mut(*index as usize).ok_or_else(invalid_index)? = value;
        }
        return Ok(data);
    }
    let source_shape = crate::indexing::plan::effective_index_shape(array.shape(), plan.dims)?;
    let target_shape = crate::indexing::plan::effective_index_shape(&plan.base_shape, plan.dims)?;
    if source_shape
        .iter()
        .enumerate()
        .any(|(dimension, extent)| target_shape.get(dimension).copied().unwrap_or(1) < *extent)
    {
        return Err(semantic_error(
            "InvalidIndexSelectorPlan",
            "Object-array assignment cannot shrink an existing dimension",
        ));
    }
    let target_len = plan.base_shape.iter().try_fold(1usize, |length, extent| {
        length.checked_mul(*extent).ok_or_else(invalid_index)
    })?;
    let old_strides = column_major_strides(&source_shape)?;
    let target_strides = column_major_strides(&target_shape)?;
    let mut slots = Vec::new();
    slots.try_reserve_exact(target_len).map_err(|_| {
        semantic_error(
            "ObjectArrayAssignment",
            "Object-array growth exceeds available storage",
        )
    })?;
    slots.resize_with(target_len, || None);
    for (index, value) in array.data().iter().enumerate() {
        let destination =
            source_shape
                .iter()
                .enumerate()
                .try_fold(0usize, |offset, (dimension, extent)| {
                    let coordinate = (index / old_strides[dimension]) % extent;
                    let stride = *target_strides.get(dimension).ok_or_else(invalid_index)?;
                    coordinate
                        .checked_mul(stride)
                        .and_then(|component| offset.checked_add(component))
                        .ok_or_else(invalid_index)
                })?;
        *slots.get_mut(destination).ok_or_else(invalid_index)? = Some(value.clone());
    }
    for (index, value) in plan.indices.iter().zip(replacements) {
        *slots.get_mut(*index as usize).ok_or_else(invalid_index)? = Some(value);
    }
    slots
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| {
                semantic_error(
                    "ObjectArrayDefaultConstructionRequired",
                    "Object-array growth with unassigned slots requires a default constructor",
                )
            })
        })
        .collect()
}

fn invalid_index() -> RuntimeError {
    semantic_error("IndexOutOfBounds", "Object array index is out of bounds")
}

#[cfg(test)]
mod tests;
