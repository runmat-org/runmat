//! Exact paired integer payloads with one independently retained shape.

use super::*;
use crate::{ComplexTensor, IntegerComplexStorage};

pub struct ExactHostIntegerComplexTensorAllocation {
    shape: SharedShape,
    real: ExactHostNumericAllocation,
    imag: ExactHostNumericAllocation,
}

impl ExactHostIntegerComplexTensorAllocation {
    pub fn requested_bytes_for_dimensions(
        dtype: NumericDType,
        len: usize,
        rank: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        if dtype.integer_class().is_none() {
            return Err(ExactHostNumericAllocationError::UnsupportedClass);
        }
        let component = ExactHostNumericAllocation::requested_bytes(dtype, len)?;
        let shape = SharedShape::requested_bytes_for_rank(rank)?;
        component
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(shape))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub const fn retained_claim_count() -> usize {
        3
    }

    pub fn new(
        dtype: NumericDType,
        shape: &[usize],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        if dtype.integer_class().is_none() {
            return Err(ExactHostNumericAllocationError::UnsupportedClass);
        }
        let len = shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let shape = SharedShape::new_admitted_from_slice(shape, permit)?;
        let real = ExactHostNumericAllocation::new(dtype, len, permit)?;
        let imag = ExactHostNumericAllocation::new(dtype, len, permit)?;
        Ok(Self { shape, real, imag })
    }

    pub fn set(
        &mut self,
        index: usize,
        real: NumericScalar,
        imag: NumericScalar,
    ) -> Result<(), String> {
        if index >= self.real.len
            || real.numeric_dtype() != self.real.dtype
            || imag.numeric_dtype() != self.real.dtype
        {
            return Err("paired integer assignment does not match exact output".into());
        }
        self.real.set_value(index, real)?;
        self.imag
            .set_value(index, imag)
            .expect("paired exact output class and length validated");
        Ok(())
    }

    pub fn into_tensor(self) -> Result<ComplexTensor, ExactHostNumericAllocationError> {
        let storage = IntegerComplexStorage::from_host_buffers(
            self.real.try_into_host_buffer()?,
            self.imag.try_into_host_buffer()?,
        )
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        ComplexTensor::from_shared_host_buffer(crate::ComplexStorage::Integer(storage), self.shape)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }
}
