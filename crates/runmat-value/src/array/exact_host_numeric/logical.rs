//! Exact logical bytes with independent payload and shape claims.

use super::{ExactHostAllocationPermit, ExactHostNumericAllocationError, RustExactRelease};
use crate::SharedShape;
use crate::{AdoptedHostAllocation, HostAllocationProvenance, HostLogicalBuffer, LogicalArray};
use std::alloc::{alloc_zeroed, Layout};
use std::borrow::Cow;
use std::ptr::NonNull;

pub struct ExactHostLogicalTensorAllocation {
    shape: SharedShape,
    allocation: AdoptedHostAllocation,
    len: usize,
}

impl ExactHostLogicalTensorAllocation {
    /// Exact backing bytes for an ordinary Rust-owned logical tensor. This
    /// excludes managed claims because the unmanaged owner retains none.
    pub fn unmanaged_requested_bytes(
        shape: &[usize],
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let len = shape
            .iter()
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Self::unmanaged_requested_bytes_for_dimensions(len, shape.len())
    }

    pub fn unmanaged_requested_bytes_for_dimensions(
        len: usize,
        rank: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        unmanaged_payload_bytes(len)?
            .checked_add(SharedShape::requested_bytes_for_rank(rank)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn requested_bytes(shape: &[usize]) -> Result<usize, ExactHostNumericAllocationError> {
        let len = shape
            .iter()
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Self::requested_bytes_for_dimensions(len, shape.len())
    }

    pub fn requested_bytes_for_dimensions(
        len: usize,
        rank: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let payload_bytes = requested_payload_bytes(len)?;
        payload_bytes
            .checked_add(SharedShape::requested_bytes_for_rank(rank)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    /// Payload and shape each retain one claim from the supplied permit.
    pub const fn retained_claim_count() -> usize {
        2
    }

    pub fn new(
        shape: &[usize],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let len = shape
            .iter()
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let layout = Layout::array::<u8>(len)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        let admitted_shape = SharedShape::new_admitted_from_slice(shape, permit)?;
        let payload_bytes = requested_payload_bytes(len)?;
        let claim = permit.reserve_exact_claim(payload_bytes)?;
        let release = RustExactRelease::try_new(layout, claim)?;
        let pointer = if len == 0 {
            NonNull::<u8>::dangling()
        } else {
            NonNull::new(unsafe { alloc_zeroed(layout) })
                .ok_or(ExactHostNumericAllocationError::AllocationFailed)?
        };
        let provenance = HostAllocationProvenance {
            allocator: Cow::Borrowed("runmat.exact-host-logical.v1"),
            byte_length: len,
            capacity_bytes: len,
            alignment: layout.align(),
        };
        let allocation = unsafe { AdoptedHostAllocation::new_owned(pointer, provenance, release) };
        Ok(Self {
            shape: admitted_shape,
            allocation,
            len,
        })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.allocation.pointer().as_ptr(), self.len) }
    }

    pub fn into_array(self) -> Result<LogicalArray, ExactHostNumericAllocationError> {
        let data = HostLogicalBuffer::try_adopt_exact(self.allocation, self.len)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
        LogicalArray::from_shared_host_buffer(data, self.shape)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }
}

fn requested_payload_bytes(len: usize) -> Result<usize, ExactHostNumericAllocationError> {
    Layout::array::<u8>(len)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
        .size()
        .checked_add(Layout::new::<RustExactRelease>().size())
        .and_then(|bytes| bytes.checked_add(HostLogicalBuffer::exact_metadata_bytes()))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}

fn unmanaged_payload_bytes(len: usize) -> Result<usize, ExactHostNumericAllocationError> {
    Layout::array::<u8>(len)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
        .size()
        .checked_add(HostLogicalBuffer::exact_metadata_bytes())
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}
