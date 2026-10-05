//! Exact-capacity numeric host storage for routes with a physical byte permit.
//! This owner allocates once and transfers that same block to a tensor.

use super::{
    AdoptedHostAllocation, HostAllocationProvenance, HostAllocationRelease, HostNumericBuffer,
    Tensor,
};
use crate::IntegerStorage;
use crate::NumericDType;
use crate::NumericScalar;
use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

mod integer;
use integer::{allocate_integer, integer_layout, ExactIntegerStorage};

macro_rules! integer_slice_mut {
    ($self:expr, $variant:ident) => {
        match &mut $self.allocation {
            ExactAllocation::Integer(allocation) => match &mut allocation.values {
                IntegerStorage::$variant(values) => Some(values.as_mut_slice()),
                _ => None,
            },
            ExactAllocation::Floating(_) => None,
        }
    };
}

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

/// Retains a quota claim through the physical output allocation's lifetime.
/// Installed exact owners can use a known-layout Box; external permits keep
/// their existing shared reservation contract.
#[derive(Debug)]
pub enum ExactHostAllocationClaim {
    Shared(Arc<dyn ExactHostAllocationReservation>),
    Owned(Box<dyn ExactHostAllocationReservation>),
}

pub trait ExactHostAllocationPermit: fmt::Debug + Send + Sync {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError>;

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        self.reserve_exact(bytes)
            .map(ExactHostAllocationClaim::Shared)
    }
}

#[derive(Debug)]
struct RustExactRelease {
    layout: Layout,
    _reservation: ExactHostAllocationClaim,
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

/// Initialized exact-capacity payload for all ten native numeric classes.
/// The caller still must account for shape, source, codec, and other live
/// allocations before claiming a complete-call finite route.
pub struct ExactHostNumericAllocation {
    allocation: ExactAllocation,
    dtype: NumericDType,
    len: usize,
}

/// One initial exact numeric payload plus its exact two-element tensor shape.
/// A single reservation covers both producer allocations before either is
/// created and remains through their initial retained ownership. A later
/// Tensor clone/reshape is a separate operation and must account for its own
/// new shape or payload; this claim does not certify those future allocations
/// or the remaining fixed Tensor/Arc metadata overhead.
pub struct ExactHostTensor2dAllocation {
    numeric: ExactHostNumericAllocation,
    shape: [usize; 2],
}

impl ExactHostTensor2dAllocation {
    /// Physical bytes requested for the initial retained tensor owner. This
    /// is the same calculation used by `new`; later COW is a separate call.
    pub fn requested_bytes(
        dtype: NumericDType,
        rows: usize,
        columns: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let count = rows
            .checked_mul(columns)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let shape_bytes = Layout::array::<usize>(2)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size();
        ExactHostNumericAllocation::requested_bytes_with_shape(dtype, count, shape_bytes)
    }

    pub fn new(
        dtype: NumericDType,
        rows: usize,
        columns: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let count = rows
            .checked_mul(columns)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let shape_bytes = Layout::array::<usize>(2)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size();
        Ok(Self {
            numeric: ExactHostNumericAllocation::new_with_reserved_shape(
                dtype,
                count,
                shape_bytes,
                permit,
            )?,
            shape: [rows, columns],
        })
    }

    pub fn into_tensor(self) -> Result<Tensor, ExactHostNumericAllocationError> {
        let layout = Layout::array::<usize>(2)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        // SAFETY: the exact two-element layout is nonzero and aligned for
        // usize. `Vec::from_raw_parts` takes the only owner with capacity two.
        let pointer = NonNull::new(unsafe { alloc(layout) })
            .ok_or(ExactHostNumericAllocationError::AllocationFailed)?
            .cast::<usize>();
        unsafe {
            pointer.as_ptr().write(self.shape[0]);
            pointer.as_ptr().add(1).write(self.shape[1]);
        }
        let shape = unsafe { Vec::from_raw_parts(pointer.as_ptr(), 2, 2) };
        Tensor::from_host_buffer(self.numeric.try_into_host_buffer()?, shape)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }
}

impl Deref for ExactHostTensor2dAllocation {
    type Target = ExactHostNumericAllocation;

    fn deref(&self) -> &Self::Target {
        &self.numeric
    }
}

impl DerefMut for ExactHostTensor2dAllocation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.numeric
    }
}

enum ExactAllocation {
    Floating(AdoptedHostAllocation),
    Integer(ExactIntegerStorage),
}

impl ExactHostNumericAllocation {
    pub fn requested_bytes(
        dtype: NumericDType,
        len: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        Self::requested_bytes_with_shape(dtype, len, 0)
    }

    fn requested_bytes_with_shape(
        dtype: NumericDType,
        len: usize,
        shape_bytes: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let layout = Self::payload_layout(dtype, len)?;
        let release_bytes = if dtype.integer_class().is_some() {
            0
        } else {
            Layout::new::<RustExactRelease>().size()
        };
        layout
            .size()
            .checked_add(shape_bytes)
            .and_then(|bytes| bytes.checked_add(release_bytes))
            .and_then(|bytes| bytes.checked_add(HostNumericBuffer::exact_metadata_bytes()))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    fn payload_layout(
        dtype: NumericDType,
        len: usize,
    ) -> Result<Layout, ExactHostNumericAllocationError> {
        match dtype {
            NumericDType::F32 => Layout::array::<f32>(len),
            NumericDType::F64 => Layout::array::<f64>(len),
            _ => integer_layout(dtype, len),
        }
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn new(
        dtype: NumericDType,
        len: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        Self::new_with_reserved_shape(dtype, len, 0, permit)
    }

    fn new_with_reserved_shape(
        dtype: NumericDType,
        len: usize,
        shape_bytes: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let layout = Self::payload_layout(dtype, len)?;
        let charged = Self::requested_bytes_with_shape(dtype, len, shape_bytes)?;
        let reservation = permit.reserve_exact_claim(charged)?;
        if dtype.integer_class().is_some() {
            return Ok(Self {
                allocation: ExactAllocation::Integer(allocate_integer(
                    dtype,
                    len,
                    layout,
                    reservation,
                )?),
                dtype,
                len,
            });
        }
        let provenance = HostAllocationProvenance {
            allocator: "runmat.exact-host-numeric.v1".into(),
            byte_length: layout.size(),
            capacity_bytes: layout.size(),
            alignment: layout.align(),
        };
        let release = Box::new(RustExactRelease {
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
        let allocation = unsafe { AdoptedHostAllocation::new_owned(pointer, provenance, release) };
        Ok(Self {
            allocation: ExactAllocation::Floating(allocation),
            dtype,
            len,
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        match &self.allocation {
            ExactAllocation::Floating(allocation) => allocation.provenance().capacity_bytes,
            ExactAllocation::Integer(allocation) => allocation.capacity_bytes(),
        }
    }

    /// Store one same-class value without replacing or cloning the admitted
    /// backing allocation. Parser callers retain exact integer bits.
    pub fn set_value(&mut self, index: usize, value: NumericScalar) -> Result<(), String> {
        if index >= self.len {
            return Err(format!(
                "numeric index {index} exceeds {} elements",
                self.len
            ));
        }
        match (&mut self.allocation, value) {
            (ExactAllocation::Floating(allocation), NumericScalar::F64(value))
                if self.dtype == NumericDType::F64 =>
            {
                // SAFETY: the constructor allocated and initialized exactly
                // `len` aligned F64 elements, and this owner is unique.
                unsafe {
                    allocation
                        .pointer()
                        .as_ptr()
                        .cast::<f64>()
                        .add(index)
                        .write(value)
                };
                Ok(())
            }
            (ExactAllocation::Floating(allocation), NumericScalar::F32(value))
                if self.dtype == NumericDType::F32 =>
            {
                // SAFETY: same ownership and checked index as the F64 arm.
                unsafe {
                    allocation
                        .pointer()
                        .as_ptr()
                        .cast::<f32>()
                        .add(index)
                        .write(value)
                };
                Ok(())
            }
            (ExactAllocation::Integer(allocation), value) => allocation.values.set_value(
                index,
                value.into_int_value().ok_or_else(|| {
                    format!(
                        "cannot store {} in {} numeric storage",
                        value.class_name(),
                        self.dtype.class_name()
                    )
                })?,
            ),
            (_, value) => Err(format!(
                "cannot store {} in {} numeric storage",
                value.class_name(),
                self.dtype.class_name()
            )),
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        (self.dtype == NumericDType::F32).then(|| {
            let ExactAllocation::Floating(allocation) = &mut self.allocation else {
                unreachable!("floating dtype has floating allocation")
            };
            // SAFETY: the owner is unique, aligned, initialized, and sized for
            // exactly `len` F32 elements until it is adopted.
            unsafe {
                std::slice::from_raw_parts_mut(allocation.pointer().as_ptr().cast(), self.len)
            }
        })
    }

    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        (self.dtype == NumericDType::F64).then(|| {
            let ExactAllocation::Floating(allocation) = &mut self.allocation else {
                unreachable!("floating dtype has floating allocation")
            };
            // SAFETY: the owner is unique, aligned, initialized, and sized for
            // exactly `len` F64 elements until it is adopted.
            unsafe {
                std::slice::from_raw_parts_mut(allocation.pointer().as_ptr().cast(), self.len)
            }
        })
    }

    pub fn as_i8_slice_mut(&mut self) -> Option<&mut [i8]> {
        integer_slice_mut!(self, I8)
    }
    pub fn as_i16_slice_mut(&mut self) -> Option<&mut [i16]> {
        integer_slice_mut!(self, I16)
    }
    pub fn as_i32_slice_mut(&mut self) -> Option<&mut [i32]> {
        integer_slice_mut!(self, I32)
    }
    pub fn as_i64_slice_mut(&mut self) -> Option<&mut [i64]> {
        integer_slice_mut!(self, I64)
    }
    pub fn as_u8_slice_mut(&mut self) -> Option<&mut [u8]> {
        integer_slice_mut!(self, U8)
    }
    pub fn as_u16_slice_mut(&mut self) -> Option<&mut [u16]> {
        integer_slice_mut!(self, U16)
    }
    pub fn as_u32_slice_mut(&mut self) -> Option<&mut [u32]> {
        integer_slice_mut!(self, U32)
    }
    pub fn as_u64_slice_mut(&mut self) -> Option<&mut [u64]> {
        integer_slice_mut!(self, U64)
    }

    pub fn into_host_buffer(self) -> HostNumericBuffer {
        match self.allocation {
            ExactAllocation::Floating(allocation) => HostNumericBuffer::try_adopt_exact(
                allocation, self.dtype, self.len,
            )
            .unwrap_or_else(|(_, detail)| unreachable!("validated exact allocation: {detail}")),
            ExactAllocation::Integer(allocation) => {
                let (storage, reservation) = allocation.into_parts();
                HostNumericBuffer::from_exact_integer_storage(storage, reservation)
            }
        }
    }

    pub fn try_into_host_buffer(
        self,
    ) -> Result<HostNumericBuffer, ExactHostNumericAllocationError> {
        match self.allocation {
            ExactAllocation::Floating(allocation) => {
                HostNumericBuffer::try_adopt_exact_fallible(allocation, self.dtype, self.len)
            }
            ExactAllocation::Integer(allocation) => {
                let (storage, reservation) = allocation.into_parts();
                HostNumericBuffer::try_from_exact_integer_storage(storage, reservation)
            }
        }
    }
}

#[cfg(test)]
mod tests;
