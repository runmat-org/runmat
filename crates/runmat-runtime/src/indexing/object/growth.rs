//! Scatter directly into one admitted final homogeneous generation.
use crate::execution::object_array::{indices::Indices, Class, ObjectArrayConstruction};
use crate::indexing::{plan::IndexPlan, shape::column_major_strides_in_context};
use crate::{runtime_error::semantic_error, RuntimeError};
use runmat_value::{ObjectArray, Value};

pub(super) fn assign(
    runtime: Option<&crate::context::RuntimeContext>,
    array: &ObjectArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let source = Value::ObjectArray(array.clone());
    let _roots = match runtime {
        Some(runtime) => crate::execution::RootedValueLease::for_values_iterator_in_context(
            [&source, rhs].into_iter(),
            runtime,
        )?,
        None => crate::execution::RootedValueLease::for_values([&source, rhs].into_iter())
            .map_err(|e| e.into_runtime_error())?,
    };
    let source_shape =
        crate::indexing::plan::effective_index_shape_in_context(runtime, array.shape(), plan.dims)?;
    let target_shape = crate::indexing::plan::effective_index_shape_in_context(
        runtime,
        &plan.base_shape,
        plan.dims,
    )?;
    if source_shape
        .iter()
        .enumerate()
        .any(|(d, &n)| target_shape.get(d).copied().unwrap_or(1) < n)
    {
        return Err(semantic_error(
            "InvalidIndexSelectorPlan",
            "Object-array assignment cannot shrink an existing dimension",
        ));
    }
    let target_len = plan
        .base_shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(invalid_index)?;
    let old_strides = column_major_strides_in_context(runtime, &source_shape)?;
    let target_strides = column_major_strides_in_context(runtime, &target_shape)?;
    let mut positions = Indices::new(runtime, target_len)?;
    for start in (0..target_len).step_by(4096) {
        positions.checkpoint()?;
        let end = start.saturating_add(4096).min(target_len);
        positions.values_mut()[start..end].fill(usize::MAX);
    }
    positions.checkpoint()?;
    for index in 0..array.len() {
        if index % 4096 == 0 {
            positions.checkpoint()?;
        }
        let destination =
            source_shape
                .iter()
                .enumerate()
                .try_fold(0usize, |offset, (d, &extent)| {
                    let coordinate = (index / old_strides[d]) % extent;
                    coordinate
                        .checked_mul(*target_strides.get(d).ok_or_else(invalid_index)?)
                        .and_then(|n| offset.checked_add(n))
                        .ok_or_else(invalid_index)
                })?;
        *positions
            .values_mut()
            .get_mut(destination)
            .ok_or_else(invalid_index)? = index;
    }
    let rhs_array = match rhs {
        Value::ObjectArray(rhs)
            if rhs.class_name() == array.class_name() && rhs.len() == plan.indices.len() =>
        {
            Some(rhs)
        }
        _ => None,
    };
    let scalar = if rhs_array.is_none() {
        Some(super::assignment_scalar(rhs, array.class_name())?)
    } else {
        None
    };
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            positions.checkpoint()?;
        }
        let encoded = array
            .len()
            .checked_add(ordinal)
            .filter(|&n| n != usize::MAX)
            .ok_or_else(invalid_index)?;
        *positions
            .values_mut()
            .get_mut(index as usize)
            .ok_or_else(invalid_index)? = encoded;
    }
    let mut output = ObjectArrayConstruction::new(
        runtime,
        Class::Owner(array.class_name_owner()),
        target_len,
        &plan.base_shape,
        array.selected_provenance().iter(),
        array.has_unselected_provenance(),
    )?;
    for (index, &position) in positions.values().iter().enumerate() {
        if index % 4096 == 0 {
            positions.checkpoint()?;
        }
        if position == usize::MAX {
            return Err(semantic_error(
                "ObjectArrayDefaultConstructionRequired",
                "Object-array growth with unassigned slots requires a default constructor",
            ));
        }
        let child = if position < array.len() {
            &array.data()[position]
        } else if let Some(rhs) = rhs_array {
            &rhs.data()[position - array.len()]
        } else {
            scalar.expect("validated scalar assignment")
        };
        output.push(child.clone())?;
    }
    let result = output.finish()?;
    positions.checkpoint()?;
    Ok(result.with_owned(|value| match value {
        Value::ObjectArray(array) if array.len() == 1 && array.shape() == [1, 1] => {
            array.data()[0].clone()
        }
        value => value,
    }))
}
fn invalid_index() -> RuntimeError {
    semantic_error("IndexOutOfBounds", "Object array index is out of bounds")
}
#[cfg(test)]
mod tests;
