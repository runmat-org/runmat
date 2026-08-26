use std::cell::UnsafeCell;
use std::ffi::c_void;

use runmat_value::Value;

use crate::{NativeLibraryMetadata, NativePointer, NativeType, PointerOwnership};

use super::arguments::{pointee_from_value, PointeeSlot};
use super::outputs::decode_pointee;
use super::InvocationError;

/// Stable, typed storage owned by a RunMat session for use with native pointer
/// arguments. Its address is never exposed as a numerical value.
pub struct NativePointerResource {
    pointee: NativeType,
    ownership: PointerOwnership,
    storage: UnsafeCell<Option<PointeeSlot>>,
}

impl std::fmt::Debug for NativePointerResource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativePointerResource")
            .field("pointee", &self.pointee)
            .field("ownership", &self.ownership)
            .finish_non_exhaustive()
    }
}

impl NativePointerResource {
    pub fn new(
        pointee: NativeType,
        initial_value: &Value,
        metadata: &NativeLibraryMetadata,
    ) -> Result<Self, InvocationError> {
        let storage = pointee_from_value("libpointer", &pointee, initial_value, metadata).map_err(
            |message| InvocationError::Argument {
                symbol: "libpointer".into(),
                argument: 2,
                name: "initial_value".into(),
                message,
            },
        )?;
        Ok(Self {
            pointee,
            ownership: PointerOwnership::CallerOwned,
            storage: UnsafeCell::new(Some(storage)),
        })
    }

    pub fn null(pointee: NativeType) -> Self {
        Self {
            pointee,
            ownership: PointerOwnership::CallerOwned,
            storage: UnsafeCell::new(None),
        }
    }

    pub fn pointee(&self) -> &NativeType {
        &self.pointee
    }

    pub fn ownership(&self) -> PointerOwnership {
        self.ownership
    }

    pub fn value(&self, metadata: &NativeLibraryMetadata) -> Result<Value, InvocationError> {
        // SAFETY: Native calls are synchronous and the session serializes use
        // of this resource. No Rust reference to its storage crosses a call.
        let storage = unsafe { &*self.storage.get() };
        let Some(storage) = storage.as_ref() else {
            return Ok(Value::Tensor(
                runmat_value::Tensor::new(Vec::new(), vec![0, 0])
                    .expect("empty pointer value has a valid shape"),
            ));
        };
        decode_pointee("libpointer", &self.pointee, storage, metadata).map_err(|message| {
            InvocationError::Output {
                symbol: "libpointer".into(),
                message,
            }
        })
    }

    pub fn set_value(
        &self,
        value: &Value,
        metadata: &NativeLibraryMetadata,
    ) -> Result<(), InvocationError> {
        let replacement = pointee_from_value("libpointer", &self.pointee, value, metadata)
            .map_err(|message| InvocationError::Argument {
                symbol: "libpointer".into(),
                argument: 2,
                name: "value".into(),
                message,
            })?;
        // SAFETY: Session serialization prevents native access while the
        // existing allocation is updated in place.
        let storage = unsafe { &mut *self.storage.get() };
        if let Some(storage) = storage {
            replace_storage(storage, replacement).map_err(|message| InvocationError::Argument {
                symbol: "libpointer".into(),
                argument: 2,
                name: "value".into(),
                message,
            })
        } else {
            *storage = Some(replacement);
            Ok(())
        }
    }

    pub(super) fn address(&self) -> *mut c_void {
        // SAFETY: The resource owns stable backing storage for its lifetime.
        // Session serialization prevents overlapping native and Rust access.
        let storage = unsafe { &mut *self.storage.get() };
        storage
            .as_mut()
            .map(PointeeSlot::address)
            .unwrap_or(std::ptr::null_mut())
    }
}

fn replace_storage(current: &mut PointeeSlot, replacement: PointeeSlot) -> Result<(), String> {
    match (current, replacement) {
        (PointeeSlot::Scalar(current), PointeeSlot::Scalar(replacement))
            if std::mem::discriminant(current) == std::mem::discriminant(&replacement) =>
        {
            *current = replacement;
            Ok(())
        }
        (
            PointeeSlot::Array { storage, shape },
            PointeeSlot::Array {
                storage: replacement,
                shape: replacement_shape,
            },
        ) if *shape == replacement_shape => replace_numeric_storage(storage, replacement),
        (PointeeSlot::Bytes(bytes), PointeeSlot::Bytes(replacement))
            if bytes.len() == replacement.len() =>
        {
            bytes.copy_from_slice(&replacement);
            Ok(())
        }
        (PointeeSlot::Structure(storage), PointeeSlot::Structure(replacement))
            if storage.bytes().len() == replacement.bytes().len() =>
        {
            storage.bytes_mut().copy_from_slice(replacement.bytes());
            Ok(())
        }
        _ => Err("replacement value changes the pointer type, shape, or allocation size".into()),
    }
}

fn replace_numeric_storage(
    current: &mut runmat_value::NumericStorage,
    replacement: runmat_value::NumericStorage,
) -> Result<(), String> {
    macro_rules! copy_variant {
        ($current:expr, $replacement:expr) => {{
            if $current.len() != $replacement.len() {
                return Err("replacement array changes the pointer allocation size".into());
            }
            $current.copy_from_slice(&$replacement);
            Ok(())
        }};
    }
    match (current, replacement) {
        (runmat_value::NumericStorage::F64(a), runmat_value::NumericStorage::F64(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::F32(a), runmat_value::NumericStorage::F32(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::I8(a), runmat_value::NumericStorage::I8(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::I16(a), runmat_value::NumericStorage::I16(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::I32(a), runmat_value::NumericStorage::I32(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::I64(a), runmat_value::NumericStorage::I64(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::U8(a), runmat_value::NumericStorage::U8(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::U16(a), runmat_value::NumericStorage::U16(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::U32(a), runmat_value::NumericStorage::U32(b)) => {
            copy_variant!(a, b)
        }
        (runmat_value::NumericStorage::U64(a), runmat_value::NumericStorage::U64(b)) => {
            copy_variant!(a, b)
        }
        _ => Err("replacement array class does not match the pointer allocation".into()),
    }
}

#[derive(Debug)]
enum PointerBindingTarget<'pointer> {
    Resource(&'pointer NativePointerResource),
    Opaque(&'pointer NativePointer),
}

#[derive(Debug)]
pub struct PointerBinding<'pointer> {
    pub argument_index: usize,
    target: PointerBindingTarget<'pointer>,
}

impl<'pointer> PointerBinding<'pointer> {
    pub fn resource(argument_index: usize, pointer: &'pointer NativePointerResource) -> Self {
        Self {
            argument_index,
            target: PointerBindingTarget::Resource(pointer),
        }
    }

    pub fn opaque(argument_index: usize, pointer: &'pointer NativePointer) -> Self {
        Self {
            argument_index,
            target: PointerBindingTarget::Opaque(pointer),
        }
    }

    pub(super) fn pointee(&self) -> &NativeType {
        match self.target {
            PointerBindingTarget::Resource(pointer) => pointer.pointee(),
            PointerBindingTarget::Opaque(pointer) => &pointer.pointee,
        }
    }

    pub(super) fn address(&self) -> *mut c_void {
        match self.target {
            PointerBindingTarget::Resource(pointer) => pointer.address(),
            PointerBindingTarget::Opaque(pointer) => pointer.address().as_ptr(),
        }
    }
}
