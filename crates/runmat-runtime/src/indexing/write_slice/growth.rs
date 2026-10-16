//! Typed host storage relocation for indexed assignment growth.
use super::{checked_assignment_element_count, map_slice_shape_error, mex, RuntimeError};
use crate::indexing::plan::effective_index_shape;
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, IntegerComplexStorage, NumericStorage, Tensor,
};

// A newly extended leading dimension changes the column-major address of
// every later column. Rebase once into a typed zero-filled result; growth of
// only the final dimension can use the storage's in-place append instead.
fn growth_restride_shape(
    old_shape: &[usize],
    new_shape: &[usize],
) -> Result<Option<Vec<usize>>, RuntimeError> {
    let old = effective_index_shape(old_shape, new_shape.len())?;
    if old.iter().zip(new_shape).any(|(old, new)| new < old) {
        return Err(mex(
            "InvalidAssignmentPlan",
            "slice assignment plan cannot shrink an addressed dimension",
        ));
    }
    Ok(old
        .iter()
        .zip(new_shape)
        .take(old.len().saturating_sub(1))
        .any(|(a, b)| a != b)
        .then_some(old))
}

fn rebased_position(mut old_index: usize, old_shape: &[usize], new_shape: &[usize]) -> usize {
    let mut new_index = 0;
    let mut new_stride = 1;
    for (&old_extent, &new_extent) in old_shape.iter().zip(new_shape) {
        let coordinate = old_index % old_extent;
        old_index /= old_extent;
        new_index += coordinate * new_stride;
        new_stride *= new_extent;
    }
    new_index
}

pub(super) fn grow_numeric_tensor(
    tensor: Tensor,
    new_shape: &[usize],
    target_len: usize,
) -> Result<NumericStorage, RuntimeError> {
    if let Some(old_shape) = growth_restride_shape(tensor.shape(), new_shape)? {
        // Borrow before consuming: a shared or adopted HostNumericBuffer would
        // otherwise materialize the entire old payload just before we allocate
        // the new, re-strided destination.
        let mut result = NumericStorage::zeros(tensor.numeric_dtype(), target_len);
        for index in 0..tensor.len() {
            let value = tensor
                .numeric_value_at(index)
                .expect("validated numeric storage");
            result
                .set_value(rebased_position(index, &old_shape, new_shape), value)
                .expect("same-class rebased numeric position is admitted");
        }
        Ok(result)
    } else {
        let mut storage = tensor
            .into_numeric_storage()
            .map_err(|error| map_slice_shape_error("slice assign", error))?;
        storage.resize_zeroed(target_len);
        Ok(storage)
    }
}

pub(super) fn grow_complex_tensor(
    mut tensor: ComplexTensor,
    new_shape: &[usize],
) -> Result<ComplexTensor, RuntimeError> {
    let target_len = checked_assignment_element_count(new_shape)?;
    if target_len < tensor.len() {
        return Err(mex(
            "InvalidAssignmentPlan",
            "slice assignment plan cannot shrink complex storage",
        ));
    }
    if tensor.shape() == new_shape {
        return Ok(tensor);
    }
    let Some(old_shape) = growth_restride_shape(tensor.shape(), new_shape)? else {
        tensor
            .resize_zeroed_for_shape(new_shape.to_vec())
            .map_err(|error| map_slice_shape_error("complex slice growth", error))?;
        return Ok(tensor);
    };
    let (storage, _) = tensor.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(values) => {
            let mut result = vec![ComplexElement(0.0, 0.0); target_len];
            for (index, value) in values.iter().copied().enumerate() {
                result[rebased_position(index, &old_shape, new_shape)] = value;
            }
            ComplexStorage::F64(result.into())
        }
        ComplexStorage::F32(values) => {
            let mut result = vec![ComplexElement(0.0, 0.0); target_len];
            for (index, value) in values.iter().copied().enumerate() {
                result[rebased_position(index, &old_shape, new_shape)] = value;
            }
            ComplexStorage::F32(result.into())
        }
        ComplexStorage::Integer(values) => {
            let mut real = values.real.zeros_like(target_len);
            let mut imag = values.imag.zeros_like(target_len);
            for index in 0..values.len() {
                let target = rebased_position(index, &old_shape, new_shape);
                real.set_value(
                    target,
                    values.real.value_at(index).expect("validated real storage"),
                )
                .expect("same-class rebased real position is admitted");
                imag.set_value(
                    target,
                    values
                        .imag
                        .value_at(index)
                        .expect("validated imaginary storage"),
                )
                .expect("same-class rebased imaginary position is admitted");
            }
            ComplexStorage::Integer(
                IntegerComplexStorage::new(real, imag)
                    .expect("paired rebased integer storage retains class and length"),
            )
        }
    };
    ComplexTensor::from_complex_storage(storage, new_shape.to_vec())
        .map_err(|error| map_slice_shape_error("complex slice growth", error))
}
