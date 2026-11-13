//! Narrow transfer seam for already-owned, already-claimed factor arrays.
use crate::{
    AdoptedHostAllocation, ExactHostNumericAllocationError, HostComplexBuffer, HostIndexBuffer,
    HostNumericBuffer, NumericDType,
};

/// No payload is allocated or copied by these transfers. The producer must
/// retain the exact quoted shared-header bytes in the allocation's release
/// authority, alongside the allocation's preexisting payload claim. This is
/// deliberately not a replacement allocator or an execution authority.
pub struct NumericFactorBufferAdoption;

impl NumericFactorBufferAdoption {
    pub const fn index_header_bytes() -> usize {
        HostIndexBuffer::exact_metadata_bytes()
    }
    pub const fn numeric_header_bytes() -> usize {
        HostNumericBuffer::exact_metadata_bytes()
    }
    pub const fn complex_f32_header_bytes() -> usize {
        HostComplexBuffer::<f32>::exact_metadata_bytes()
    }
    pub const fn complex_f64_header_bytes() -> usize {
        HostComplexBuffer::<f64>::exact_metadata_bytes()
    }

    pub fn indices(
        allocation: AdoptedHostAllocation,
        len: usize,
    ) -> Result<HostIndexBuffer, ExactHostNumericAllocationError> {
        HostIndexBuffer::try_adopt_exact_fallible(allocation, len)
    }
    pub fn numeric(
        allocation: AdoptedHostAllocation,
        dtype: NumericDType,
        len: usize,
    ) -> Result<HostNumericBuffer, ExactHostNumericAllocationError> {
        HostNumericBuffer::try_adopt_exact_fallible(allocation, dtype, len)
    }
    pub fn complex_f32(
        allocation: AdoptedHostAllocation,
        len: usize,
    ) -> Result<HostComplexBuffer<f32>, ExactHostNumericAllocationError> {
        HostComplexBuffer::try_adopt_exact_fallible(allocation, len)
    }
    pub fn complex_f64(
        allocation: AdoptedHostAllocation,
        len: usize,
    ) -> Result<HostComplexBuffer<f64>, ExactHostNumericAllocationError> {
        HostComplexBuffer::try_adopt_exact_fallible(allocation, len)
    }
}
