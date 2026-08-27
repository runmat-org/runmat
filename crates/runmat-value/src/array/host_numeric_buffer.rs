use crate::{IntegerStorage, NumericDType, NumericScalar, NumericStorage};
use std::ffi::c_void;

use super::host_allocation::HostData;
use super::host_buffer::HostBuffer;
use super::{record_host_copy, AdoptedHostAllocation, HostCopyReason};

#[derive(Debug, Clone, PartialEq)]
enum HostNumericStorage {
    F64(HostData<f64>),
    F32(HostData<f32>),
    Integer(IntegerStorage),
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

impl HostNumericBuffer {
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
            | NumericStorage::U64(_)) => HostNumericStorage::Integer(
                storage
                    .into_integer_storage()
                    .expect("integer NumericStorage variant"),
            ),
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
        macro_rules! adopt {
            ($variant:ident, $type:ty) => {
                HostData::<$type>::try_adopt(allocation, len).map(HostNumericStorage::$variant)
            };
        }
        let storage = match dtype {
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
                return Err((
                    allocation,
                    "adopted integer storage requires the typed integer-view migration".into(),
                ));
            }
        }?;
        Ok(Self {
            storage: HostBuffer::new(storage),
        })
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        match self.storage.get() {
            HostNumericStorage::F64(_) => NumericDType::F64,
            HostNumericStorage::F32(_) => NumericDType::F32,
            HostNumericStorage::Integer(storage) => storage.numeric_dtype(),
        }
    }

    pub fn len(&self) -> usize {
        match self.storage.get() {
            HostNumericStorage::F64(values) => values.as_slice().len(),
            HostNumericStorage::F32(values) => values.as_slice().len(),
            HostNumericStorage::Integer(storage) => storage.len(),
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
            HostNumericStorage::Integer(storage) => Some(storage),
            _ => None,
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
                storage.value_at(index).map(NumericScalar::from)
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
            HostNumericStorage::Integer(storage) => storage.to_f64_vec(),
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
        match self.storage.make_mut() {
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
                let value = match value {
                    NumericScalar::F64(value) => value as f32,
                    NumericScalar::F32(value) => value,
                    value => value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer")
                        .to_f64() as f32,
                };
                values[index] = value;
            }
            HostNumericStorage::Integer(storage) => {
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
            HostNumericStorage::Integer(storage) => resize_integer_zeroed(storage, len),
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
            HostNumericStorage::Integer(storage) => integer_data_pointer(storage),
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
            HostNumericStorage::Integer(storage) => integer_data_pointer_mut(storage),
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
            HostNumericStorage::Integer(storage) => NumericStorage::from_integer_storage(storage),
        }
    }
}

fn storage_is_adopted(storage: &HostNumericStorage) -> bool {
    match storage {
        HostNumericStorage::F64(values) => values.is_adopted(),
        HostNumericStorage::F32(values) => values.is_adopted(),
        HostNumericStorage::Integer(_) => false,
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
            copy_integer_storage(destination, source)
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
