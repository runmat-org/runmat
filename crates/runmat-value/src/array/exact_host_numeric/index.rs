//! Exact native-width indices shared by sparse topology and factor maps.

use super::*;
use crate::HostIndexBuffer;

/// An initialized, uniquely writable index allocation. Publication transfers
/// the same payload and its physical reservation to the canonical buffer.
pub struct ExactHostIndexBufferAllocation {
    allocation: AdoptedHostAllocation,
    len: usize,
}

impl ExactHostIndexBufferAllocation {
    pub const fn retained_claim_count() -> usize {
        1
    }

    pub fn requested_bytes(len: usize) -> Result<usize, ExactHostNumericAllocationError> {
        Layout::array::<usize>(len)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size()
            .checked_add(Layout::new::<RustExactRelease>().size())
            .and_then(|bytes| bytes.checked_add(HostIndexBuffer::exact_metadata_bytes()))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn new(
        len: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let layout = Layout::array::<usize>(len)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        let claim = permit.reserve_exact_claim(Self::requested_bytes(len)?)?;
        let release = RustExactRelease::try_new(layout, claim)?;
        let pointer = if layout.size() == 0 {
            NonNull::<usize>::dangling().cast()
        } else {
            // SAFETY: the admitted layout is nonzero and every zeroed bit
            // pattern is a valid usize. RustExactRelease owns its deallocation.
            NonNull::new(unsafe { alloc_zeroed(layout) })
                .ok_or(ExactHostNumericAllocationError::AllocationFailed)?
        };
        let provenance = HostAllocationProvenance {
            allocator: Cow::Borrowed("runmat.exact-host-sparse-index.v1"),
            byte_length: layout.size(),
            capacity_bytes: layout.size(),
            alignment: layout.align(),
        };
        // SAFETY: the payload is unique, initialized, correctly aligned and
        // released exactly once by the matching layout owner.
        let allocation = unsafe { AdoptedHostAllocation::new_owned(pointer, provenance, release) };
        Ok(Self { allocation, len })
    }

    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        // SAFETY: this builder exclusively owns len initialized usize values,
        // including an aligned dangling pointer for an empty allocation.
        unsafe {
            std::slice::from_raw_parts_mut(self.allocation.pointer().as_ptr().cast(), self.len)
        }
    }

    pub fn into_buffer(self) -> Result<HostIndexBuffer, String> {
        HostIndexBuffer::try_adopt_exact(self.allocation, self.len)
    }

    /// Typed no-copy publication, preserving physical allocation failures.
    pub fn into_buffer_fallible(self) -> Result<HostIndexBuffer, ExactHostNumericAllocationError> {
        HostIndexBuffer::try_adopt_exact_fallible(self.allocation, self.len)
    }
}
