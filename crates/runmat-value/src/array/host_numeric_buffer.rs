use crate::{IntegerStorage, NumericDType, NumericScalar, NumericStorage};
use std::ffi::c_void;

use super::host_allocation::HostData;
use super::host_buffer::HostBuffer;
use super::{
    record_host_copy, AdoptedHostAllocation, ExactHostNumericAllocationError, HostCopyReason,
};

#[derive(Debug)]
struct OwnedIntegerStorage {
    values: IntegerStorage,
    // Only the original exact allocation carries its quota claim. Copy-on-
    // write creates a distinct ordinary value in a subsequent operation.
    _reservation: Option<super::ExactHostAllocationClaim>,
}

impl Clone for OwnedIntegerStorage {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            _reservation: None,
        }
    }
}

impl PartialEq for OwnedIntegerStorage {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

#[derive(Debug, Clone, PartialEq)]
enum HostNumericStorage {
    F64(HostData<f64>),
    F32(HostData<f32>),
    Integer(OwnedIntegerStorage),
}

/// Pointer-stable, copy-on-write host storage for a dense numeric value.
///
/// Cloning this handle retains the allocation rather than copying its payload.
/// Safe mutation uses copy-on-write so ordinary RunMat value semantics remain
/// unchanged. Foreign runtimes may borrow the allocation only for the lifetime
/// of an invocation lease; callers of [`Self::foreign_data_pointer`] must uphold
/// that contract.
#[derive(Debug, Clone, PartialEq)]
pub struct HostNumericBuffer {
    storage: HostBuffer<HostNumericStorage>,
}

/// Non-growing unique borrow of the canonical numeric allocation. Its original
/// physical claim remains on the borrowed owner; this never performs COW.
pub struct UniqueHostNumericMutation<'a> {
    storage: &'a mut HostNumericStorage,
}
impl UniqueHostNumericMutation<'_> {
    pub fn set_value(&mut self, index: usize, value: NumericScalar) -> Result<(), String> {
        assign_storage_value(self.storage, index, value)
    }
}

fn assign_storage_value(
    storage: &mut HostNumericStorage,
    index: usize,
    value: NumericScalar,
) -> Result<(), String> {
    let len = match &*storage {
        HostNumericStorage::F64(values) => values.len(),
        HostNumericStorage::F32(values) => values.len(),
        HostNumericStorage::Integer(storage) => storage.values.len(),
    };
    if index >= len {
        return Err(format!("numeric buffer index {index} out of bounds"));
    }
    match storage {
        HostNumericStorage::F64(values) => {
            let value = match value {
                NumericScalar::F64(value) => value,
                NumericScalar::F32(value) => f64::from(value),
                value => value
                    .into_int_value()
                    .expect("non-floating numeric scalar is integer")
                    .to_f64(),
            };
            values[index] = value;
        }
        HostNumericStorage::F32(values) => {
            values[index] = value.materialize_f32();
        }
        HostNumericStorage::Integer(storage) => {
            let storage = &mut storage.values;
            let exact = match value {
                NumericScalar::F64(value) => storage.cast_f64_assignment(value),
                NumericScalar::F32(value) => storage.cast_f64_assignment(f64::from(value)),
                value => storage.cast_exact_assignment(
                    &value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer"),
                ),
            };
            storage.set_value(index, exact)?;
        }
    }
    Ok(())
}

impl HostNumericBuffer {
    /// Acquire the existing allocation uniquely or decline without copying.
    pub fn try_unique_mutation(&mut self) -> Option<UniqueHostNumericMutation<'_>> {
        self.storage
            .try_get_mut_unique()
            .map(|storage| UniqueHostNumericMutation { storage })
    }
    pub(crate) const fn exact_metadata_bytes() -> usize {
        HostBuffer::<HostNumericStorage>::exact_metadata_bytes()
    }

    pub fn from_numeric_storage(storage: NumericStorage) -> Self {
        let storage = match storage {
            NumericStorage::F64(values) => HostNumericStorage::F64(HostData::from_vec(values)),
            NumericStorage::F32(values) => HostNumericStorage::F32(HostData::from_vec(values)),
            storage @ (NumericStorage::I8(_)
            | NumericStorage::I16(_)
            | NumericStorage::I32(_)
            | NumericStorage::I64(_)
            | NumericStorage::U8(_)
            | NumericStorage::U16(_)
            | NumericStorage::U32(_)
            | NumericStorage::U64(_)) => HostNumericStorage::Integer(OwnedIntegerStorage {
                values: storage
                    .into_integer_storage()
                    .expect("integer NumericStorage variant"),
                _reservation: None,
            }),
        };
        Self {
            storage: HostBuffer::new(storage),
        }
    }

    pub fn try_adopt(
        allocation: AdoptedHostAllocation,
        dtype: NumericDType,
        len: usize,
    ) -> Result<Self, (AdoptedHostAllocation, String)> {
        Self::try_adopt_with_owner(allocation, dtype, len, false)
    }

    pub(crate) fn try_adopt_exact(
        allocation: AdoptedHostAllocation,
        dtype: NumericDType,
        len: usize,
    ) -> Result<Self, (AdoptedHostAllocation, String)> {
        Self::try_adopt_with_owner(allocation, dtype, len, true)
    }

    pub(crate) fn try_adopt_exact_fallible(
        allocation: AdoptedHostAllocation,
        dtype: NumericDType,
        len: usize,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let storage = Self::try_adopt_storage(allocation, dtype, len)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        HostBuffer::try_new_exact(storage)
            .map(|storage| Self { storage })
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    fn try_adopt_with_owner(
        allocation: AdoptedHostAllocation,
        dtype: NumericDType,
        len: usize,
        exact: bool,
    ) -> Result<Self, (AdoptedHostAllocation, String)> {
        let storage = Self::try_adopt_storage(allocation, dtype, len)?;
        Ok(Self {
            storage: if exact {
                HostBuffer::new_exact(storage)
            } else {
                HostBuffer::new(storage)
            },
        })
    }

    fn try_adopt_storage(
        allocation: AdoptedHostAllocation,
        dtype: NumericDType,
        len: usize,
    ) -> Result<HostNumericStorage, (AdoptedHostAllocation, String)> {
        macro_rules! adopt {
            ($variant:ident, $type:ty) => {
                HostData::<$type>::try_adopt(allocation, len).map(HostNumericStorage::$variant)
            };
        }
        match dtype {
            NumericDType::F64 => adopt!(F64, f64),
            NumericDType::F32 => adopt!(F32, f32),
            NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64 => {
                Err((
                    allocation,
                    "external adopted integer storage is unavailable; exact integer allocations use the owned typed constructor".into(),
                ))
            }
        }
    }

    pub(crate) fn from_exact_integer_storage(
        values: IntegerStorage,
        reservation: super::ExactHostAllocationClaim,
    ) -> Self {
        Self {
            storage: HostBuffer::new_exact(HostNumericStorage::Integer(OwnedIntegerStorage {
                values,
                _reservation: Some(reservation),
            })),
        }
    }

    pub(crate) fn try_from_exact_integer_storage(
        values: IntegerStorage,
        reservation: super::ExactHostAllocationClaim,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        HostBuffer::try_new_exact(HostNumericStorage::Integer(OwnedIntegerStorage {
            values,
            _reservation: Some(reservation),
        }))
        .map(|storage| Self { storage })
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        match self.storage.get() {
            HostNumericStorage::F64(_) => NumericDType::F64,
            HostNumericStorage::F32(_) => NumericDType::F32,
            HostNumericStorage::Integer(storage) => storage.values.numeric_dtype(),
        }
    }

    pub fn len(&self) -> usize {
        match self.storage.get() {
            HostNumericStorage::F64(values) => values.as_slice().len(),
            HostNumericStorage::F32(values) => values.as_slice().len(),
            HostNumericStorage::Integer(storage) => storage.values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn checked_byte_len(&self) -> Option<usize> {
        self.len().checked_mul(self.numeric_dtype().byte_size())
    }

    pub fn integer_storage(&self) -> Option<&IntegerStorage> {
        match self.storage.get() {
            HostNumericStorage::Integer(storage) => Some(&storage.values),
            _ => None,
        }
    }

    pub(crate) fn has_retained_allocation(&self) -> bool {
        storage_is_adopted(self.storage.get())
    }

    pub(crate) fn retained_integer_is_uniquely_mutable(&self) -> bool {
        self.has_retained_allocation()
            && !self.storage.is_shared()
            && matches!(self.storage.get(), HostNumericStorage::Integer(_))
    }

    pub(crate) fn retained_integer_can_resize_without_reallocation(&self, len: usize) -> bool {
        self.retained_integer_is_uniquely_mutable()
            && self
                .integer_storage()
                .is_some_and(|storage| len <= integer_storage_capacity(storage))
    }

    /// Mutates an exact integer backing only when this handle is its unique
    /// owner. Unlike ordinary copy-on-write mutation, this never detaches the
    /// payload from its retained-allocation claim.
    pub(crate) fn set_exact_integer_value_unique(
        &mut self,
        index: usize,
        value: NumericScalar,
    ) -> Result<(), String> {
        let storage = self
            .storage
            .try_get_mut_unique()
            .ok_or_else(|| "retained integer component is still aliased".to_string())?;
        let HostNumericStorage::Integer(storage) = storage else {
            return Err("retained integer component has a non-integer class".into());
        };
        if storage._reservation.is_none() {
            return Err("integer component has no retained allocation".into());
        }
        let exact = match value {
            NumericScalar::F64(value) => storage.values.cast_f64_assignment(value),
            NumericScalar::F32(value) => storage.values.cast_f64_assignment(f64::from(value)),
            value => storage.values.cast_exact_assignment(
                &value
                    .into_int_value()
                    .expect("non-floating numeric scalar is integer"),
            ),
        };
        storage.values.set_value(index, exact)
    }

    /// Resizes an exact integer backing without moving its physical block.
    /// Growth beyond the originally claimed Vec capacity is rejected.
    pub(crate) fn resize_exact_integer_zeroed_unique(&mut self, len: usize) -> Result<(), String> {
        let storage = self
            .storage
            .try_get_mut_unique()
            .ok_or_else(|| "retained integer component is still aliased".to_string())?;
        let HostNumericStorage::Integer(storage) = storage else {
            return Err("retained integer component has a non-integer class".into());
        };
        if storage._reservation.is_none() {
            return Err("integer component has no retained allocation".into());
        }
        if len > integer_storage_capacity(&storage.values) {
            return Err(
                "retained integer component cannot grow beyond its admitted capacity".into(),
            );
        }
        resize_integer_zeroed(&mut storage.values, len);
        Ok(())
    }

    /// Borrow a unique native payload without implicit copy-on-write.
    pub fn try_as_f64_slice_mut_unique(&mut self) -> Option<&mut [f64]> {
        match self.storage.try_get_mut_unique()? {
            HostNumericStorage::F64(values) => Some(values.as_mut_slice()),
            HostNumericStorage::F32(_) | HostNumericStorage::Integer(_) => None,
        }
    }

    /// Borrow a unique native payload without implicit copy-on-write.
    pub fn try_as_f32_slice_mut_unique(&mut self) -> Option<&mut [f32]> {
        match self.storage.try_get_mut_unique()? {
            HostNumericStorage::F32(values) => Some(values.as_mut_slice()),
            HostNumericStorage::F64(_) | HostNumericStorage::Integer(_) => None,
        }
    }

    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self.storage.get() {
            HostNumericStorage::F64(values) => Some(values.as_slice()),
            _ => None,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self.storage.get() {
            HostNumericStorage::F32(values) => Some(values.as_slice()),
            _ => None,
        }
    }

    pub fn value_at(&self, index: usize) -> Option<NumericScalar> {
        match self.storage.get() {
            HostNumericStorage::F64(values) => values.get(index).copied().map(NumericScalar::F64),
            HostNumericStorage::F32(values) => values.get(index).copied().map(NumericScalar::F32),
            HostNumericStorage::Integer(storage) => {
                storage.values.value_at(index).map(NumericScalar::from)
            }
        }
    }

    pub fn materialize_f64(&self) -> Vec<f64> {
        record_host_copy(
            HostCopyReason::OwnedMaterialization,
            self.len().saturating_mul(std::mem::size_of::<f64>()),
        );
        match self.storage.get() {
            HostNumericStorage::F64(values) => values.to_vec(),
            HostNumericStorage::F32(values) => values.iter().copied().map(f64::from).collect(),
            HostNumericStorage::Integer(storage) => storage.values.to_f64_vec(),
        }
    }

    pub fn set_value(&mut self, index: usize, value: NumericScalar) -> Result<(), String> {
        if index >= self.len() {
            return Err(format!("numeric buffer index {index} out of bounds"));
        }
        if self.storage.is_shared() {
            record_host_copy(
                HostCopyReason::CopyOnWriteMutation,
                self.checked_byte_len()
                    .expect("allocated numeric buffer byte length fits usize"),
            );
        }
        assign_storage_value(self.storage.make_mut(), index, value)
    }

    pub fn copy_from_same_type(&mut self, source: &Self) -> Result<(), String> {
        if self.numeric_dtype() != source.numeric_dtype() || self.len() != source.len() {
            return Err("numeric buffer class or length does not match".into());
        }
        if self.storage.is_shared() {
            record_host_copy(
                HostCopyReason::CopyOnWriteMutation,
                self.checked_byte_len()
                    .expect("allocated numeric buffer byte length fits usize"),
            );
        }
        copy_storage(self.storage.make_mut(), source.storage.get());
        Ok(())
    }

    pub fn resize_zeroed(&mut self, len: usize) {
        if self.storage.is_shared() || storage_is_adopted(self.storage.get()) {
            record_host_copy(
                HostCopyReason::CopyOnWriteMutation,
                self.checked_byte_len()
                    .expect("allocated numeric buffer byte length fits usize"),
            );
        }
        match self.storage.make_mut() {
            HostNumericStorage::F64(values) => values.ensure_rust_vec().resize(len, 0.0),
            HostNumericStorage::F32(values) => values.ensure_rust_vec().resize(len, 0.0),
            HostNumericStorage::Integer(storage) => {
                if storage._reservation.is_some() {
                    // Detach before any operation that can resize or move the
                    // originally reserved block.
                    let mut detached = storage.values.clone();
                    resize_integer_zeroed(&mut detached, len);
                    storage.values = detached;
                    storage._reservation = None;
                } else {
                    resize_integer_zeroed(&mut storage.values, len);
                }
            }
        }
    }

    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        self.storage.shares_allocation_with(&other.storage)
    }

    pub fn make_unique(&mut self) {
        if self.storage.is_shared() {
            record_host_copy(
                HostCopyReason::CopyOnWriteMutation,
                self.checked_byte_len()
                    .expect("allocated numeric buffer byte length fits usize"),
            );
        }
        self.storage.make_unique();
    }

    /// Returns the stable address of this allocation for a synchronous foreign
    /// invocation. The pointer is read-only from RunMat's perspective even
    /// though compatibility APIs may expose a mutable C pointer.
    ///
    /// # Safety
    ///
    /// No Rust reference into this allocation may be used while foreign code is
    /// executing. Foreign code must not mutate a logically read-only input, and
    /// the pointer must not outlive the handle or its invocation lease.
    pub unsafe fn foreign_data_pointer(&self) -> *mut c_void {
        match self.storage.get() {
            HostNumericStorage::F64(values) => values.as_ptr().cast_mut().cast(),
            HostNumericStorage::F32(values) => values.as_ptr().cast_mut().cast(),
            HostNumericStorage::Integer(storage) => integer_data_pointer(&storage.values),
        }
    }

    /// Returns a writable pointer after applying ordinary copy-on-write value
    /// semantics. This is used for host-owned foreign outputs and structural
    /// replacement, never for borrowed input arrays.
    ///
    /// # Safety
    ///
    /// The pointer must remain invocation-scoped and the caller must not create
    /// overlapping Rust references while it is used.
    pub unsafe fn foreign_data_pointer_mut(&mut self) -> *mut c_void {
        if self.storage.is_shared() {
            record_host_copy(
                HostCopyReason::CopyOnWriteMutation,
                self.checked_byte_len()
                    .expect("allocated numeric buffer byte length fits usize"),
            );
        }
        match self.storage.make_mut() {
            HostNumericStorage::F64(values) => values.as_mut_ptr().cast(),
            HostNumericStorage::F32(values) => values.as_mut_ptr().cast(),
            HostNumericStorage::Integer(storage) => integer_data_pointer_mut(&mut storage.values),
        }
    }

    pub fn into_numeric_storage(self) -> NumericStorage {
        if self.storage.is_shared() || storage_is_adopted(self.storage.get()) {
            record_host_copy(
                HostCopyReason::OwnedMaterialization,
                self.checked_byte_len()
                    .expect("allocated numeric buffer byte length fits usize"),
            );
        }
        match self.storage.into_inner() {
            HostNumericStorage::F64(values) => NumericStorage::F64(values.into_vec()),
            HostNumericStorage::F32(values) => NumericStorage::F32(values.into_vec()),
            HostNumericStorage::Integer(storage) => {
                NumericStorage::from_integer_storage(if storage._reservation.is_some() {
                    storage.values.clone()
                } else {
                    storage.values
                })
            }
        }
    }
}

fn storage_is_adopted(storage: &HostNumericStorage) -> bool {
    match storage {
        HostNumericStorage::F64(values) => values.is_adopted(),
        HostNumericStorage::F32(values) => values.is_adopted(),
        HostNumericStorage::Integer(storage) => storage._reservation.is_some(),
    }
}

fn integer_storage_capacity(storage: &IntegerStorage) -> usize {
    match storage {
        IntegerStorage::I8(values) => values.capacity(),
        IntegerStorage::I16(values) => values.capacity(),
        IntegerStorage::I32(values) => values.capacity(),
        IntegerStorage::I64(values) => values.capacity(),
        IntegerStorage::U8(values) => values.capacity(),
        IntegerStorage::U16(values) => values.capacity(),
        IntegerStorage::U32(values) => values.capacity(),
        IntegerStorage::U64(values) => values.capacity(),
    }
}

fn copy_storage(destination: &mut HostNumericStorage, source: &HostNumericStorage) {
    match (destination, source) {
        (HostNumericStorage::F64(destination), HostNumericStorage::F64(source)) => {
            destination.copy_from_slice(source)
        }
        (HostNumericStorage::F32(destination), HostNumericStorage::F32(source)) => {
            destination.copy_from_slice(source)
        }
        (HostNumericStorage::Integer(destination), HostNumericStorage::Integer(source)) => {
            copy_integer_storage(&mut destination.values, &source.values)
        }
        _ => unreachable!("numeric dtype equality guarantees matching host storage"),
    }
}

fn copy_integer_storage(destination: &mut IntegerStorage, source: &IntegerStorage) {
    match (destination, source) {
        (IntegerStorage::I8(destination), IntegerStorage::I8(source)) => {
            destination.copy_from_slice(source)
        }
        (IntegerStorage::I16(destination), IntegerStorage::I16(source)) => {
            destination.copy_from_slice(source)
        }
        (IntegerStorage::I32(destination), IntegerStorage::I32(source)) => {
            destination.copy_from_slice(source)
        }
        (IntegerStorage::I64(destination), IntegerStorage::I64(source)) => {
            destination.copy_from_slice(source)
        }
        (IntegerStorage::U8(destination), IntegerStorage::U8(source)) => {
            destination.copy_from_slice(source)
        }
        (IntegerStorage::U16(destination), IntegerStorage::U16(source)) => {
            destination.copy_from_slice(source)
        }
        (IntegerStorage::U32(destination), IntegerStorage::U32(source)) => {
            destination.copy_from_slice(source)
        }
        (IntegerStorage::U64(destination), IntegerStorage::U64(source)) => {
            destination.copy_from_slice(source)
        }
        _ => unreachable!("numeric dtype equality guarantees matching integer storage"),
    }
}

fn resize_integer_zeroed(storage: &mut IntegerStorage, len: usize) {
    match storage {
        IntegerStorage::I8(values) => values.resize(len, 0),
        IntegerStorage::I16(values) => values.resize(len, 0),
        IntegerStorage::I32(values) => values.resize(len, 0),
        IntegerStorage::I64(values) => values.resize(len, 0),
        IntegerStorage::U8(values) => values.resize(len, 0),
        IntegerStorage::U16(values) => values.resize(len, 0),
        IntegerStorage::U32(values) => values.resize(len, 0),
        IntegerStorage::U64(values) => values.resize(len, 0),
    }
}

fn integer_data_pointer(storage: &IntegerStorage) -> *mut c_void {
    match storage {
        IntegerStorage::I8(values) => values.as_ptr().cast_mut().cast(),
        IntegerStorage::I16(values) => values.as_ptr().cast_mut().cast(),
        IntegerStorage::I32(values) => values.as_ptr().cast_mut().cast(),
        IntegerStorage::I64(values) => values.as_ptr().cast_mut().cast(),
        IntegerStorage::U8(values) => values.as_ptr().cast_mut().cast(),
        IntegerStorage::U16(values) => values.as_ptr().cast_mut().cast(),
        IntegerStorage::U32(values) => values.as_ptr().cast_mut().cast(),
        IntegerStorage::U64(values) => values.as_ptr().cast_mut().cast(),
    }
}

fn integer_data_pointer_mut(storage: &mut IntegerStorage) -> *mut c_void {
    match storage {
        IntegerStorage::I8(values) => values.as_mut_ptr().cast(),
        IntegerStorage::I16(values) => values.as_mut_ptr().cast(),
        IntegerStorage::I32(values) => values.as_mut_ptr().cast(),
        IntegerStorage::I64(values) => values.as_mut_ptr().cast(),
        IntegerStorage::U8(values) => values.as_mut_ptr().cast(),
        IntegerStorage::U16(values) => values.as_mut_ptr().cast(),
        IntegerStorage::U32(values) => values.as_mut_ptr().cast(),
        IntegerStorage::U64(values) => values.as_mut_ptr().cast(),
    }
}
