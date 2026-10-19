//! No-copy adoption of producer-owned native numeric Vec backings.

use super::*;
use crate::{
    ComplexElement, ComplexStorage, ComplexTensor, HostComplexBuffer, HostLogicalBuffer,
    LogicalArray,
};
use std::alloc::{alloc, dealloc};
use std::fmt;
use std::marker::PhantomData;

trait NativeVecElement: Copy + fmt::Debug + Send + Sync + 'static {}

impl NativeVecElement for f32 {}
impl NativeVecElement for f64 {}
impl NativeVecElement for u8 {}

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
    metadata_bytes: usize,
) -> Result<usize, ExactHostNumericAllocationError> {
    Layout::array::<T>(capacity)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
        .size()
        .checked_add(Layout::new::<RustVecRelease<T>>().size())
        .and_then(|bytes| bytes.checked_add(metadata_bytes))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}

fn adopt_raw_allocation<T: NativeVecElement>(
    values: Vec<T>,
    permit: &dyn ExactHostAllocationPermit,
    metadata_bytes: usize,
) -> Result<AdoptedHostAllocation, ExactHostNumericAllocationError> {
    let len = values.len();
    let capacity = values.capacity();
    let byte_length = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
    let capacity_layout = Layout::array::<T>(capacity)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
    let claim =
        permit.reserve_exact_claim(payload_requested_bytes::<T>(capacity, metadata_bytes)?)?;
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
    Ok(allocation)
}

fn adopt_payload<T: NativeVecElement>(
    values: Vec<T>,
    dtype: NumericDType,
    permit: &dyn ExactHostAllocationPermit,
) -> Result<HostNumericBuffer, ExactHostNumericAllocationError> {
    let len = values.len();
    let allocation =
        adopt_raw_allocation(values, permit, HostNumericBuffer::exact_metadata_bytes())?;
    HostNumericBuffer::try_adopt_exact_fallible(allocation, dtype, len)
}

fn adopt_tensor<T: NativeVecElement>(
    values: Vec<T>,
    shape: Vec<usize>,
    dtype: NumericDType,
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
    let storage = adopt_payload(values, dtype, permit)?;
    Tensor::from_shared_host_buffer(storage, shape)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
}

fn integer_payload_requested_bytes<T>(
    capacity: usize,
) -> Result<usize, ExactHostNumericAllocationError> {
    Layout::array::<T>(capacity)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
        .size()
        .checked_add(HostNumericBuffer::exact_metadata_bytes())
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}

fn adopt_integer_tensor<T>(
    values: Vec<T>,
    shape: Vec<usize>,
    permit: &dyn ExactHostAllocationPermit,
    wrap: impl FnOnce(Vec<T>) -> IntegerStorage,
) -> Result<Tensor, ExactHostNumericAllocationError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
    if expected != values.len() {
        return Err(ExactHostNumericAllocationError::LayoutOverflow);
    }
    let shape = SharedShape::new_admitted_from_owned(shape, permit)?;
    let payload_bytes = integer_payload_requested_bytes::<T>(values.capacity())?;
    let claim = permit.reserve_exact_claim(payload_bytes)?;
    let storage = HostNumericBuffer::try_from_exact_integer_storage(wrap(values), claim)?;
    Tensor::from_shared_host_buffer(storage, shape)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
}

macro_rules! integer_adopters {
    ($(($dtype:ident, $element:ty, $method:ident)),+ $(,)?) => {
        impl ExactHostTensorAllocation {
            $(
                pub fn $method(
                    values: Vec<$element>,
                    shape: Vec<usize>,
                    permit: &dyn ExactHostAllocationPermit,
                ) -> Result<Tensor, ExactHostNumericAllocationError> {
                    adopt_integer_tensor(values, shape, permit, IntegerStorage::$dtype)
                }
            )+
        }
    };
}

integer_adopters!(
    (I8, i8, adopt_i8_vec),
    (I16, i16, adopt_i16_vec),
    (I32, i32, adopt_i32_vec),
    (I64, i64, adopt_i64_vec),
    (U8, u8, adopt_u8_vec),
    (U16, u16, adopt_u16_vec),
    (U32, u32, adopt_u32_vec),
    (U64, u64, adopt_u64_vec),
);

impl ExactHostTensorAllocation {
    /// Exact physical bytes retained when adopting existing Rust Vec payload
    /// and shape allocations with these actual capacities.
    pub fn adopted_requested_bytes(
        dtype: NumericDType,
        value_capacity: usize,
        shape_capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let payload = match dtype {
            NumericDType::F32 => payload_requested_bytes::<f32>(
                value_capacity,
                HostNumericBuffer::exact_metadata_bytes(),
            )?,
            NumericDType::F64 => payload_requested_bytes::<f64>(
                value_capacity,
                HostNumericBuffer::exact_metadata_bytes(),
            )?,
            NumericDType::I8 => integer_payload_requested_bytes::<i8>(value_capacity)?,
            NumericDType::I16 => integer_payload_requested_bytes::<i16>(value_capacity)?,
            NumericDType::I32 => integer_payload_requested_bytes::<i32>(value_capacity)?,
            NumericDType::I64 => integer_payload_requested_bytes::<i64>(value_capacity)?,
            NumericDType::U8 => integer_payload_requested_bytes::<u8>(value_capacity)?,
            NumericDType::U16 => integer_payload_requested_bytes::<u16>(value_capacity)?,
            NumericDType::U32 => integer_payload_requested_bytes::<u32>(value_capacity)?,
            NumericDType::U64 => integer_payload_requested_bytes::<u64>(value_capacity)?,
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
        adopt_tensor(values, shape, NumericDType::F32, permit)
    }

    /// Move one F64 payload and shape Vec into canonical tensor ownership
    /// without copying or replacing either backing allocation.
    pub fn adopt_f64_vec(
        values: Vec<f64>,
        shape: Vec<usize>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Tensor, ExactHostNumericAllocationError> {
        adopt_tensor(values, shape, NumericDType::F64, permit)
    }
}

impl ExactHostLogicalTensorAllocation {
    /// Physical retained bytes for adopting canonical zero/one provider bytes.
    pub fn adopted_requested_bytes(
        value_capacity: usize,
        shape_capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        payload_requested_bytes::<u8>(value_capacity, HostLogicalBuffer::exact_metadata_bytes())?
            .checked_add(SharedShape::requested_bytes_for_capacity(shape_capacity)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn adopt_u8_vec(
        values: Vec<u8>,
        shape: Vec<usize>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<LogicalArray, ExactHostNumericAllocationError> {
        let expected = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        if expected != values.len() || values.iter().any(|value| *value > 1) {
            return Err(ExactHostNumericAllocationError::LayoutOverflow);
        }
        let shape = SharedShape::new_admitted_from_owned(shape, permit)?;
        let len = values.len();
        let allocation =
            adopt_raw_allocation(values, permit, HostLogicalBuffer::exact_metadata_bytes())?;
        let storage = HostLogicalBuffer::try_adopt_exact(allocation, len)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
        LogicalArray::from_shared_host_buffer(storage, shape)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
    }
}

fn adopt_complex<T: NativeVecElement>(
    values: Vec<T>,
    shape: Vec<usize>,
    permit: &dyn ExactHostAllocationPermit,
    wrap: impl FnOnce(
        AdoptedHostAllocation,
        usize,
    ) -> Result<ComplexStorage, ExactHostNumericAllocationError>,
    metadata_bytes: usize,
) -> Result<ComplexTensor, ExactHostNumericAllocationError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
    if expected.checked_mul(2) != Some(values.len())
        || std::mem::size_of::<ComplexElement<T>>() != 2 * std::mem::size_of::<T>()
        || std::mem::align_of::<ComplexElement<T>>() != std::mem::align_of::<T>()
    {
        return Err(ExactHostNumericAllocationError::LayoutOverflow);
    }
    let shape = SharedShape::new_admitted_from_owned(shape, permit)?;
    let allocation = adopt_raw_allocation(values, permit, metadata_bytes)?;
    let storage = wrap(allocation, expected)?;
    ComplexTensor::from_shared_host_buffer(storage, shape)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)
}

impl ExactHostComplexTensorAllocation {
    pub fn adopted_requested_bytes(
        dtype: NumericDType,
        value_capacity: usize,
        shape_capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let payload = match dtype {
            NumericDType::F32 => payload_requested_bytes::<f32>(
                value_capacity,
                HostComplexBuffer::<f32>::exact_metadata_bytes(),
            )?,
            NumericDType::F64 => payload_requested_bytes::<f64>(
                value_capacity,
                HostComplexBuffer::<f64>::exact_metadata_bytes(),
            )?,
            _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
        };
        payload
            .checked_add(SharedShape::requested_bytes_for_capacity(shape_capacity)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn adopt_f32_interleaved_vec(
        values: Vec<f32>,
        shape: Vec<usize>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<ComplexTensor, ExactHostNumericAllocationError> {
        adopt_complex(
            values,
            shape,
            permit,
            |allocation, len| {
                HostComplexBuffer::<f32>::try_adopt_exact(allocation, len)
                    .map(ComplexStorage::F32)
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
            },
            HostComplexBuffer::<f32>::exact_metadata_bytes(),
        )
    }

    pub fn adopt_f64_interleaved_vec(
        values: Vec<f64>,
        shape: Vec<usize>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<ComplexTensor, ExactHostNumericAllocationError> {
        adopt_complex(
            values,
            shape,
            permit,
            |allocation, len| {
                HostComplexBuffer::<f64>::try_adopt_exact(allocation, len)
                    .map(ComplexStorage::F64)
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
            },
            HostComplexBuffer::<f64>::exact_metadata_bytes(),
        )
    }
}
