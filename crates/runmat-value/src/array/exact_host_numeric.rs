//! Exact-capacity floating host storage for routes with a physical byte permit.
//! This owner allocates once and transfers that same block to a tensor.

use super::{
    AdoptedHostAllocation, HostAllocationProvenance, HostAllocationRelease, HostNumericBuffer,
};
use crate::NumericDType;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExactHostNumericAllocationError {
    UnsupportedClass,
    LayoutOverflow,
    ExceedsPermit,
    AllocationFailed,
}

impl fmt::Display for ExactHostNumericAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ExactHostNumericAllocationError {}

/// A route-owned reservation must remain live for the entire output value
/// lifetime, not merely until the decoder returns. The service implements
/// atomic quota accounting and revocation; this allocation owner never
/// invents a machine budget.
pub trait ExactHostAllocationReservation: fmt::Debug + Send + Sync {}

pub trait ExactHostAllocationPermit: fmt::Debug + Send + Sync {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError>;
}

#[derive(Debug)]
struct RustExactRelease {
    layout: Layout,
    _reservation: Arc<dyn ExactHostAllocationReservation>,
}

impl HostAllocationRelease for RustExactRelease {
    unsafe fn release(&self, pointer: NonNull<u8>) {
        if self.layout.size() != 0 {
            // SAFETY: this is the original pointer and layout allocated by
            // `ExactHostNumericAllocation::new`, and ownership transfers once.
            unsafe { dealloc(pointer.as_ptr(), self.layout) };
        }
    }
}

/// Initialized exact-capacity F32/F64 payload. The caller still must account
/// for shape, source, codec, and other live allocations before claiming a
/// complete-call finite route. Integer adoption is not yet supported.
pub struct ExactHostNumericAllocation {
    allocation: AdoptedHostAllocation,
    dtype: NumericDType,
    len: usize,
}

impl ExactHostNumericAllocation {
    pub fn new(
        dtype: NumericDType,
        len: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let layout = match dtype {
            NumericDType::F32 => Layout::array::<f32>(len),
            NumericDType::F64 => Layout::array::<f64>(len),
            _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
        }
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        let reservation = permit.reserve_exact(layout.size())?;
        let provenance = HostAllocationProvenance {
            allocator: "runmat.exact-host-numeric.v1".into(),
            byte_length: layout.size(),
            capacity_bytes: layout.size(),
            alignment: layout.align(),
        };
        let release = Arc::new(RustExactRelease {
            layout,
            _reservation: reservation,
        });
        let pointer = if layout.size() == 0 {
            match dtype {
                NumericDType::F32 => NonNull::<f32>::dangling().cast(),
                NumericDType::F64 => NonNull::<f64>::dangling().cast(),
                _ => unreachable!("class admitted above"),
            }
        } else {
            // SAFETY: `layout` is checked and nonzero. The allocation is
            // immediately owned by `AdoptedHostAllocation` below.
            NonNull::new(unsafe { alloc_zeroed(layout) })
                .ok_or(ExactHostNumericAllocationError::AllocationFailed)?
        };
        // SAFETY: the pointer has exactly the recorded layout and the release
        // authority owns that layout until the final host buffer is dropped.
        let allocation = unsafe { AdoptedHostAllocation::new(pointer, provenance, release) };
        Ok(Self {
            allocation,
            dtype,
            len,
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.allocation.provenance().capacity_bytes
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        (self.dtype == NumericDType::F32).then(|| {
            // SAFETY: the owner is unique, aligned, initialized, and sized for
            // exactly `len` F32 elements until it is adopted.
            unsafe {
                std::slice::from_raw_parts_mut(self.allocation.pointer().as_ptr().cast(), self.len)
            }
        })
    }

    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        (self.dtype == NumericDType::F64).then(|| {
            // SAFETY: the owner is unique, aligned, initialized, and sized for
            // exactly `len` F64 elements until it is adopted.
            unsafe {
                std::slice::from_raw_parts_mut(self.allocation.pointer().as_ptr().cast(), self.len)
            }
        })
    }

    pub fn into_host_buffer(self) -> HostNumericBuffer {
        HostNumericBuffer::try_adopt(self.allocation, self.dtype, self.len)
            .unwrap_or_else(|(_, detail)| unreachable!("validated exact allocation: {detail}"))
    }
}

#[cfg(test)]
mod tests;
