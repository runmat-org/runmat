//! No-copy adoption of producer-owned native floating Vec backings.

use super::*;
use std::alloc::{alloc, dealloc};
use std::fmt;
use std::marker::PhantomData;

trait NativeVecElement: Copy + fmt::Debug + Send + Sync + 'static {
    const DTYPE: NumericDType;
}

impl NativeVecElement for f32 {
    const DTYPE: NumericDType = NumericDType::F32;
}

impl NativeVecElement for f64 {
    const DTYPE: NumericDType = NumericDType::F64;
}

#[derive(Debug)]
struct RustVecRelease<T: NativeVecElement> {
    len: usize,
    capacity: usize,
    marker: PhantomData<T>,
    _claim: ExactHostAllocationClaim,
}

impl<T: NativeVecElement> RustVecRelease<T> {
    fn try_new(
        len: usize,
        capacity: usize,
        claim: ExactHostAllocationClaim,
    ) -> Result<Box<Self>, ExactHostNumericAllocationError> {
        // SAFETY: this allocates exactly one known-layout release owner. The
        // Box is constructed only after the complete value is initialized.
        let pointer = unsafe { alloc(Layout::new::<Self>()) }.cast::<Self>();
        if pointer.is_null() {
            return Err(ExactHostNumericAllocationError::AllocationFailed);
        }
        unsafe {
            pointer.write(Self {
                len,
                capacity,
                marker: PhantomData,
                _claim: claim,
            });
            Ok(Box::from_raw(pointer))
        }
    }
}

impl<T: NativeVecElement> HostAllocationRelease for RustVecRelease<T> {
    unsafe fn release(&self, _: NonNull<u8>) {
        unreachable!("Rust Vec adoption always transfers its unique release owner")
    }

    unsafe fn release_owned(self: Box<Self>, pointer: NonNull<u8>) {
        let owner = Box::into_raw(self);
        // SAFETY: consuming the unique Box makes all owner fields unique.
        let len = unsafe { (*owner).len };
        let capacity = unsafe { (*owner).capacity };
        let claim = unsafe { std::ptr::read(std::ptr::addr_of!((*owner)._claim)) };
        // SAFETY: adoption transferred this exact Vec allocation and its
        // original length/capacity. Dropping it releases elements/backing.
        drop(unsafe { Vec::from_raw_parts(pointer.as_ptr().cast::<T>(), len, capacity) });
        // SAFETY: the pointer came from the exact Box layout above.
        unsafe { dealloc(owner.cast::<u8>(), Layout::new::<Self>()) };
        drop(claim);
    }
}

fn payload_requested_bytes<T: NativeVecElement>(
    capacity: usize,
) -> Result<usize, ExactHostNumericAllocationError> {
    Layout::array::<T>(capacity)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
        .size()
        .checked_add(Layout::new::<RustVecRelease<T>>().size())
        .and_then(|bytes| bytes.checked_add(HostNumericBuffer::exact_metadata_bytes()))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}

fn adopt_payload<T: NativeVecElement>(
    values: Vec<T>,
    permit: &dyn ExactHostAllocationPermit,
) -> Result<HostNumericBuffer, ExactHostNumericAllocationError> {
    let len = values.len();
    let capacity = values.capacity();
    let byte_length = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
    let capacity_layout = Layout::array::<T>(capacity)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
    let claim = permit.reserve_exact_claim(payload_requested_bytes::<T>(capacity)?)?;
    let release = RustVecRelease::<T>::try_new(len, capacity, claim)?;
    let mut values = ManuallyDrop::new(values);
    let pointer = NonNull::new(values.as_mut_ptr().cast::<u8>())
        .expect("Vec pointers are non-null, including zero-capacity Vecs");
    let allocation = unsafe {
        AdoptedHostAllocation::new_owned(
            pointer,
            HostAllocationProvenance {
                allocator: "rust-vec".into(),
                byte_length,
                capacity_bytes: capacity_layout.size(),
                alignment: capacity_layout.align(),
            },
            release,
        )
    };
    HostNumericBuffer::try_adopt_exact_fallible(allocation, T::DTYPE, len)
}

fn adopt_tensor<T: NativeVecElement>(
    values: Vec<T>,
    shape: Vec<usize>,
    permit: &dyn ExactHostAllocationPermit,
) -> Result<Tensor, ExactHostNumericAllocationError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
    if expected != values.len() {
        return Err(ExactHostNumericAllocationError::LayoutOverflow);
    }
    let shape = SharedShape::new_admitted_from_owned(shape, permit)?;
    let storage = adopt_payload(values, permit)?;
    Tensor::from_shared_host_buffer(storage, shape)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
}

impl ExactHostTensorAllocation {
    /// Exact physical bytes retained when adopting existing Rust Vec payload
    /// and shape allocations with these actual capacities.
    pub fn adopted_requested_bytes(
        dtype: NumericDType,
        value_capacity: usize,
        shape_capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let payload = match dtype {
            NumericDType::F32 => payload_requested_bytes::<f32>(value_capacity)?,
            NumericDType::F64 => payload_requested_bytes::<f64>(value_capacity)?,
            _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
        };
        payload
            .checked_add(SharedShape::requested_bytes_for_capacity(shape_capacity)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    /// Move one F32 payload and shape Vec into canonical tensor ownership
    /// without copying or replacing either backing allocation.
    pub fn adopt_f32_vec(
        values: Vec<f32>,
        shape: Vec<usize>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Tensor, ExactHostNumericAllocationError> {
        adopt_tensor(values, shape, permit)
    }

    /// Move one F64 payload and shape Vec into canonical tensor ownership
    /// without copying or replacing either backing allocation.
    pub fn adopt_f64_vec(
        values: Vec<f64>,
        shape: Vec<usize>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Tensor, ExactHostNumericAllocationError> {
        adopt_tensor(values, shape, permit)
    }
}
