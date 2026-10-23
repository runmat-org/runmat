//! Exact interleaved complex output with independent payload and shape claims.

use super::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, RustExactRelease, SharedShape,
};
use crate::{
    AdoptedHostAllocation, ComplexElement, ComplexStorage, ComplexTensor, HostAllocationProvenance,
    HostComplexBuffer, NumericDType,
};
use std::alloc::{alloc_zeroed, Layout};
use std::borrow::Cow;
use std::ptr::NonNull;

pub struct ExactHostComplexTensorAllocation {
    shape: SharedShape,
    allocation: AdoptedHostAllocation,
    dtype: NumericDType,
    len: usize,
}

impl ExactHostComplexTensorAllocation {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        self.dtype
    }

    /// Exact backing bytes for an ordinary Rust-owned complex tensor. No
    /// release authority or retained-claim header exists on this path.
    pub fn unmanaged_requested_bytes(
        dtype: NumericDType,
        shape: &[usize],
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Self::unmanaged_requested_bytes_for_dimensions(dtype, count, shape.len())
    }

    pub fn unmanaged_requested_bytes_for_dimensions(
        dtype: NumericDType,
        count: usize,
        rank: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        unmanaged_payload_bytes(dtype, count)?
            .checked_add(SharedShape::requested_bytes_for_rank(rank)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

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
        let payload_bytes = requested_payload_bytes(dtype, count)?;
        payload_bytes
            .checked_add(SharedShape::requested_bytes_for_rank(rank)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    /// Payload and shape each retain one claim from the supplied permit.
    pub const fn retained_claim_count() -> usize {
        2
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
        let admitted_shape = SharedShape::new_admitted_from_slice(shape, permit)?;
        let payload_bytes = requested_payload_bytes(dtype, len)?;
        let claim = permit.reserve_exact_claim(payload_bytes)?;
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
            shape: admitted_shape,
            allocation,
            dtype,
            len,
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

    /// Write a native single-precision element without a floating conversion,
    /// preserving NaN payloads and signed zero in portable decoder ingress.
    pub fn set_f32(
        &mut self,
        index: usize,
        value: ComplexElement<f32>,
    ) -> Result<(), ExactHostNumericAllocationError> {
        if self.dtype != NumericDType::F32 {
            return Err(ExactHostNumericAllocationError::UnsupportedClass);
        }
        if index >= self.len {
            return Err(ExactHostNumericAllocationError::LayoutOverflow);
        }
        // SAFETY: the checked native class and index identify one initialized
        // aligned element in this uniquely owned exact allocation.
        unsafe {
            self.allocation
                .pointer()
                .as_ptr()
                .cast::<ComplexElement<f32>>()
                .add(index)
                .write(value);
        }
        Ok(())
    }

    /// Borrow uniquely owned, initialized native single-complex output.
    pub fn as_complex_f32_slice_mut(&mut self) -> Option<&mut [ComplexElement<f32>]> {
        if self.dtype != NumericDType::F32 {
            return None;
        }
        // SAFETY: new checked this class, allocation layout and length, and
        // zero-initialized every element. The mutable borrow excludes aliases
        // and cannot outlive this owner or its allocation/retained claim.
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                self.allocation
                    .pointer()
                    .as_ptr()
                    .cast::<ComplexElement<f32>>(),
                self.len,
            )
        })
    }
    /// Borrow uniquely owned, initialized native double-complex output.
    pub fn as_complex_f64_slice_mut(&mut self) -> Option<&mut [ComplexElement<f64>]> {
        if self.dtype != NumericDType::F64 {
            return None;
        }
        // SAFETY: same exact ownership/layout invariant as the single view;
        // the class gate prevents reinterpreting single storage as double.
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                self.allocation
                    .pointer()
                    .as_ptr()
                    .cast::<ComplexElement<f64>>(),
                self.len,
            )
        })
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
        ComplexTensor::from_shared_host_buffer(storage, self.shape)
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

fn requested_payload_bytes(
    dtype: NumericDType,
    count: usize,
) -> Result<usize, ExactHostNumericAllocationError> {
    let metadata = match dtype {
        NumericDType::F32 => HostComplexBuffer::<f32>::exact_metadata_bytes(),
        NumericDType::F64 => HostComplexBuffer::<f64>::exact_metadata_bytes(),
        _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
    };
    payload_layout(dtype, count)?
        .size()
        .checked_add(Layout::new::<RustExactRelease>().size())
        .and_then(|bytes| bytes.checked_add(metadata))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}

fn unmanaged_payload_bytes(
    dtype: NumericDType,
    count: usize,
) -> Result<usize, ExactHostNumericAllocationError> {
    let metadata = match dtype {
        NumericDType::F32 => HostComplexBuffer::<f32>::exact_metadata_bytes(),
        NumericDType::F64 => HostComplexBuffer::<f64>::exact_metadata_bytes(),
        _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
    };
    payload_layout(dtype, count)?
        .size()
        .checked_add(metadata)
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}
