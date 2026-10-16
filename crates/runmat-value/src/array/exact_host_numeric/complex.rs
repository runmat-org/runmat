//! Exact interleaved complex output with one retained physical quota claim.

use super::{
    exact_shape, ExactHostAllocationPermit, ExactHostNumericAllocationError, RustExactRelease,
};
use crate::{
    AdoptedHostAllocation, ComplexElement, ComplexStorage, ComplexTensor, HostAllocationProvenance,
    HostComplexBuffer, NumericDType,
};
use std::alloc::{alloc_zeroed, Layout};
use std::borrow::Cow;
use std::ptr::NonNull;

pub struct ExactHostComplexTensorAllocation {
    allocation: AdoptedHostAllocation,
    dtype: NumericDType,
    len: usize,
    shape: Vec<usize>,
}

impl ExactHostComplexTensorAllocation {
    pub fn requested_bytes(
        dtype: NumericDType,
        shape: &[usize],
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Self::requested_bytes_for_dimensions(dtype, count, shape.len())
    }

    pub fn requested_bytes_for_dimensions(
        dtype: NumericDType,
        count: usize,
        rank: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let payload = payload_layout(dtype, count)?;
        let metadata = match dtype {
            NumericDType::F32 => HostComplexBuffer::<f32>::exact_metadata_bytes(),
            NumericDType::F64 => HostComplexBuffer::<f64>::exact_metadata_bytes(),
            _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
        };
        payload
            .size()
            .checked_add(Layout::new::<RustExactRelease>().size())
            .and_then(|bytes| bytes.checked_add(metadata))
            .and_then(|bytes| bytes.checked_add(Layout::array::<usize>(rank).ok()?.size()))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn new(
        dtype: NumericDType,
        shape: &[usize],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let len = shape
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let layout = payload_layout(dtype, len)?;
        let charged = Self::requested_bytes(dtype, shape)?;
        let claim = permit.reserve_exact_claim(charged)?;
        let release = RustExactRelease::try_new(layout, claim)?;
        let pointer = if layout.size() == 0 {
            match dtype {
                NumericDType::F32 => NonNull::<ComplexElement<f32>>::dangling().cast(),
                NumericDType::F64 => NonNull::<ComplexElement<f64>>::dangling().cast(),
                _ => unreachable!("admitted floating class"),
            }
        } else {
            NonNull::new(unsafe { alloc_zeroed(layout) })
                .ok_or(ExactHostNumericAllocationError::AllocationFailed)?
        };
        let provenance = HostAllocationProvenance {
            allocator: Cow::Borrowed("runmat.exact-host-complex.v1"),
            byte_length: layout.size(),
            capacity_bytes: layout.size(),
            alignment: layout.align(),
        };
        let allocation = unsafe { AdoptedHostAllocation::new_owned(pointer, provenance, release) };
        Ok(Self {
            allocation,
            dtype,
            len,
            shape: exact_shape(shape)?,
        })
    }

    pub fn set(
        &mut self,
        index: usize,
        real: f64,
        imaginary: f64,
    ) -> Result<(), ExactHostNumericAllocationError> {
        if index >= self.len {
            return Err(ExactHostNumericAllocationError::LayoutOverflow);
        }
        unsafe {
            match self.dtype {
                NumericDType::F32 => self
                    .allocation
                    .pointer()
                    .as_ptr()
                    .cast::<ComplexElement<f32>>()
                    .add(index)
                    .write(ComplexElement(real as f32, imaginary as f32)),
                NumericDType::F64 => self
                    .allocation
                    .pointer()
                    .as_ptr()
                    .cast::<ComplexElement<f64>>()
                    .add(index)
                    .write(ComplexElement(real, imaginary)),
                _ => unreachable!("admitted floating class"),
            }
        }
        Ok(())
    }

    pub fn into_tensor(self) -> Result<ComplexTensor, ExactHostNumericAllocationError> {
        let storage = match self.dtype {
            NumericDType::F32 => ComplexStorage::F32(
                HostComplexBuffer::try_adopt_exact(self.allocation, self.len)
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?,
            ),
            NumericDType::F64 => ComplexStorage::F64(
                HostComplexBuffer::try_adopt_exact(self.allocation, self.len)
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?,
            ),
            _ => unreachable!("admitted floating class"),
        };
        ComplexTensor::from_complex_storage(storage, self.shape)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }
}

fn payload_layout(
    dtype: NumericDType,
    count: usize,
) -> Result<Layout, ExactHostNumericAllocationError> {
    match dtype {
        NumericDType::F32 => Layout::array::<ComplexElement<f32>>(count),
        NumericDType::F64 => Layout::array::<ComplexElement<f64>>(count),
        _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
    }
    .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
}
