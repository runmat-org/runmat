//! Exact-capacity numeric host storage for routes with a physical byte permit.
//! This owner allocates once and transfers that same block to a tensor.

use super::{
    AdoptedHostAllocation, HostAllocationProvenance, HostAllocationRelease, HostNumericBuffer,
    SharedShape, Tensor,
};
use crate::IntegerStorage;
use crate::NumericDType;
use crate::NumericScalar;
use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::borrow::Cow;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

mod adopted_vec;
mod complex;
mod index;
mod integer;
mod integer_complex;
mod logical;
mod sparse;
pub use complex::ExactHostComplexTensorAllocation;
pub use index::ExactHostIndexBufferAllocation;
use integer::{allocate_integer, integer_layout, ExactIntegerStorage};
pub use integer_complex::ExactHostIntegerComplexTensorAllocation;
pub use logical::ExactHostLogicalTensorAllocation;
pub use sparse::{
    ExactHostSparseClass, ExactHostSparsePublicationError, ExactHostSparseTensorAllocation,
    PreparedExactSparseTensor,
};

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

macro_rules! integer_slice {
    ($self:expr, $variant:ident) => {
        match &$self.allocation {
            ExactAllocation::Integer(allocation) => match &allocation.values {
                IntegerStorage::$variant(values) => Some(values.as_slice()),
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
pub trait ExactHostAllocationReservation:
    runmat_execution::resource::RetainedHostAllocationClaim
{
    /// Retire a uniquely owned claim. Implementations whose admitted bytes
    /// include their Box backing override this method so that backing is
    /// released before the physical reservation is refunded.
    fn retire_owned(self: Box<Self>) {
        drop(self);
    }
}

/// Retains a quota claim through the physical output allocation's lifetime.
/// Installed exact owners can use a known-layout Box; external permits keep
/// their existing shared reservation contract.
#[derive(Debug)]
pub enum ExactHostAllocationClaim {
    Shared(Arc<dyn ExactHostAllocationReservation>),
    Owned(ManuallyDrop<Box<dyn ExactHostAllocationReservation>>),
}

impl ExactHostAllocationClaim {
    pub fn owned(claim: Box<dyn ExactHostAllocationReservation>) -> Self {
        Self::Owned(ManuallyDrop::new(claim))
    }
}

impl Drop for ExactHostAllocationClaim {
    fn drop(&mut self) {
        if let Self::Owned(claim) = self {
            // SAFETY: the field is taken only from this enum's unique Drop.
            let claim = unsafe { ManuallyDrop::take(claim) };
            claim.retire_owned();
        }
    }
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

impl RustExactRelease {
    fn try_new(
        layout: Layout,
        reservation: ExactHostAllocationClaim,
    ) -> Result<Box<Self>, ExactHostNumericAllocationError> {
        let allocate = |layout| {
            // SAFETY: the exact global-allocator block is released by Box.
            unsafe { alloc(layout) }
        };
        // SAFETY: `alloc` returns a unique block of exactly the requested
        // layout or null; Box takes sole ownership after initialization.
        unsafe { Self::try_new_with(layout, reservation, allocate) }
    }

    /// # Safety
    /// A non-null allocator result must be a fresh, uniquely owned global
    /// allocation with exactly the supplied layout and alignment.
    unsafe fn try_new_with(
        layout: Layout,
        reservation: ExactHostAllocationClaim,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<Box<Self>, ExactHostNumericAllocationError> {
        let pointer = allocate(Layout::new::<Self>()).cast::<Self>();
        if pointer.is_null() {
            return Err(ExactHostNumericAllocationError::AllocationFailed);
        }
        // SAFETY: the pointer uniquely owns one exact-layout block and the
        // initialized release value transfers directly into Box ownership.
        unsafe {
            pointer.write(Self {
                layout,
                _reservation: reservation,
            });
            Ok(Box::from_raw(pointer))
        }
    }

    /// Retire the numeric block and this known-layout release header before
    /// ending the retained allocation claim that funded both.
    ///
    /// # Safety
    /// Both callbacks must release the supplied exact global-allocator block
    /// once and must not unwind after ownership has transferred from its Box.
    unsafe fn release_owned_with(
        owner: Box<Self>,
        pointer: NonNull<u8>,
        deallocate_payload: impl FnOnce(NonNull<u8>, Layout),
        deallocate_owner: impl FnOnce(NonNull<u8>, Layout),
    ) {
        let owner = Box::into_raw(owner);
        // SAFETY: consuming the sole Box makes both fields uniquely owned.
        let layout = unsafe { (*owner).layout };
        let reservation = unsafe { std::ptr::read(std::ptr::addr_of!((*owner)._reservation)) };
        if layout.size() != 0 {
            deallocate_payload(pointer, layout);
        }
        // SAFETY: `owner` came from a Box<Self> with this exact layout.
        deallocate_owner(
            unsafe { NonNull::new_unchecked(owner.cast::<u8>()) },
            Layout::new::<Self>(),
        );
        drop(reservation);
    }
}

impl HostAllocationRelease for RustExactRelease {
    unsafe fn release(&self, pointer: NonNull<u8>) {
        if self.layout.size() != 0 {
            // SAFETY: this is the original pointer and layout allocated by
            // `ExactHostNumericAllocation::new`, and ownership transfers once.
            unsafe { dealloc(pointer.as_ptr(), self.layout) };
        }
    }

    unsafe fn release_owned(self: Box<Self>, pointer: NonNull<u8>) {
        // SAFETY: both pointers retain their exact original global-allocator
        // layouts and this consumes the unique release authority.
        unsafe {
            Self::release_owned_with(
                self,
                pointer,
                |pointer, layout| dealloc(pointer.as_ptr(), layout),
                |pointer, layout| dealloc(pointer.as_ptr(), layout),
            )
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

/// One initial exact numeric payload plus its independently admitted shape.
/// The two retained allocations take distinct claims from the same permit, so
/// sharing or replacing either owner never transfers the other's quota claim.
pub struct ExactHostTensor2dAllocation {
    shape: SharedShape,
    numeric: ExactHostNumericAllocation,
}

/// Exact floating payload and exact initial shape for an arbitrary-rank
/// retained tensor. Payload and shape retain independent physical claims.
pub struct ExactHostTensorAllocation {
    shape: SharedShape,
    numeric: ExactHostNumericAllocation,
}

impl ExactHostTensorAllocation {
    /// Borrow the immutable dimensions owned by this exact allocation.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Native class of the exact payload.
    pub fn numeric_dtype(&self) -> NumericDType {
        self.numeric.dtype
    }

    /// Exact backing bytes for an ordinary Rust-owned tensor constructed
    /// without retained allocation claims. This includes the typed payload,
    /// its shared host owner, and the independently shared shape owner.
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
        ExactHostNumericAllocation::unmanaged_requested_bytes(dtype, count)?
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
        ExactHostNumericAllocation::requested_bytes(dtype, count)?
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
        let count = shape
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let admitted_shape = SharedShape::new_admitted_from_slice(shape, permit)?;
        let numeric = ExactHostNumericAllocation::new(dtype, count, permit)?;
        Ok(Self {
            shape: admitted_shape,
            numeric,
        })
    }

    pub fn into_tensor(self) -> Result<Tensor, ExactHostNumericAllocationError> {
        Tensor::from_shared_host_buffer(self.numeric.try_into_host_buffer()?, self.shape)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }
}

impl Deref for ExactHostTensorAllocation {
    type Target = ExactHostNumericAllocation;
    fn deref(&self) -> &Self::Target {
        &self.numeric
    }
}

impl DerefMut for ExactHostTensorAllocation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.numeric
    }
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
        ExactHostNumericAllocation::requested_bytes(dtype, count)?
            .checked_add(SharedShape::requested_bytes_for_rank(2)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    /// Payload and shape each retain one claim from the supplied permit.
    pub const fn retained_claim_count() -> usize {
        2
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
        let shape = [rows, columns];
        let admitted_shape = SharedShape::new_admitted_from_slice(&shape, permit)?;
        let numeric = ExactHostNumericAllocation::new(dtype, count, permit)?;
        Ok(Self {
            shape: admitted_shape,
            numeric,
        })
    }

    pub fn into_tensor(self) -> Result<Tensor, ExactHostNumericAllocationError> {
        Tensor::from_shared_host_buffer(self.numeric.try_into_host_buffer()?, self.shape)
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
    /// One retained claim owns this numeric buffer and its release metadata.
    /// Tensor shape storage has its own independent claim.
    pub const fn retained_claim_count() -> usize {
        1
    }

    fn unmanaged_requested_bytes(
        dtype: NumericDType,
        len: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        Self::payload_layout(dtype, len)?
            .size()
            .checked_add(HostNumericBuffer::exact_metadata_bytes())
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn requested_bytes(
        dtype: NumericDType,
        len: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let layout = Self::payload_layout(dtype, len)?;
        let release_bytes = if dtype.integer_class().is_some() {
            0
        } else {
            Layout::new::<RustExactRelease>().size()
        };
        layout
            .size()
            .checked_add(release_bytes)
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
        let layout = Self::payload_layout(dtype, len)?;
        let charged = Self::requested_bytes(dtype, len)?;
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
            allocator: Cow::Borrowed("runmat.exact-host-numeric.v1"),
            byte_length: layout.size(),
            capacity_bytes: layout.size(),
            alignment: layout.align(),
        };
        let release = RustExactRelease::try_new(layout, reservation)?;
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

    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        (self.dtype == NumericDType::F64).then(|| {
            let ExactAllocation::Floating(allocation) = &self.allocation else {
                unreachable!("floating dtype has floating allocation")
            };
            unsafe { std::slice::from_raw_parts(allocation.pointer().as_ptr().cast(), self.len) }
        })
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        (self.dtype == NumericDType::F32).then(|| {
            let ExactAllocation::Floating(allocation) = &self.allocation else {
                unreachable!("floating dtype has floating allocation")
            };
            unsafe { std::slice::from_raw_parts(allocation.pointer().as_ptr().cast(), self.len) }
        })
    }

    pub fn as_i8_slice(&self) -> Option<&[i8]> {
        integer_slice!(self, I8)
    }
    pub fn as_i16_slice(&self) -> Option<&[i16]> {
        integer_slice!(self, I16)
    }
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        integer_slice!(self, I32)
    }
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        integer_slice!(self, I64)
    }
    pub fn as_u8_slice(&self) -> Option<&[u8]> {
        integer_slice!(self, U8)
    }
    pub fn as_u16_slice(&self) -> Option<&[u16]> {
        integer_slice!(self, U16)
    }
    pub fn as_u32_slice(&self) -> Option<&[u32]> {
        integer_slice!(self, U32)
    }
    pub fn as_u64_slice(&self) -> Option<&[u64]> {
        integer_slice!(self, U64)
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
