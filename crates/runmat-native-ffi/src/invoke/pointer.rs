use std::cell::UnsafeCell;
use std::ffi::c_void;

use runmat_value::Value;

use crate::{NativeLibraryMetadata, NativePointer, NativeScalar, NativeType, PointerOwnership};

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

    pub fn is_null(&self) -> bool {
        // SAFETY: Reading whether the session-owned optional allocation is
        // present does not expose or alias its contents.
        unsafe { (&*self.storage.get()).is_none() }
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

/// Copy a caller-declared view of an opaque native pointer into a RunMat
/// value. The copy is bounded and keeps raw memory access inside this crate.
pub fn copy_pointer_value(
    pointer: &NativePointer,
    pointee: &NativeType,
    shape: &[usize],
    metadata: &NativeLibraryMetadata,
) -> Result<Value, InvocationError> {
    const MAX_COPY_BYTES: usize = 512 * 1024 * 1024;

    if shape.is_empty() || shape.contains(&0) {
        return Err(pointer_output_error(
            "pointer views require one or more positive dimensions",
        ));
    }
    let element_count = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
    let Some(element_count) = element_count else {
        return Err(pointer_output_error(
            "pointer view dimensions overflow usize",
        ));
    };
    if pointer.is_null() {
        return Err(pointer_output_error(
            "a null pointer cannot be copied into a nonempty value",
        ));
    }

    match pointee {
        NativeType::Scalar { scalar }
        | NativeType::Enumeration {
            storage: scalar, ..
        } => copy_scalar_pointer(pointer, *scalar, element_count, shape, MAX_COPY_BYTES),
        NativeType::Structure { .. } if element_count == 1 => {
            let (size, _) = super::abi::type_layout("lib.pointer", pointee, metadata)?;
            if size > MAX_COPY_BYTES {
                return Err(pointer_output_error("pointer view exceeds the copy limit"));
            }
            let bytes = copy_address(pointer, size);
            super::outputs::decode_pointer_structure(pointee, &bytes, metadata)
        }
        NativeType::Structure { .. } => Err(pointer_output_error(
            "arrays of structures require an explicit prepared copy contract",
        )),
        _ => Err(pointer_output_error(
            "pointer view type is not supported for direct copying",
        )),
    }
}

fn copy_scalar_pointer(
    pointer: &NativePointer,
    scalar: NativeScalar,
    element_count: usize,
    shape: &[usize],
    max_bytes: usize,
) -> Result<Value, InvocationError> {
    macro_rules! copy_numeric {
        ($native:ty, $variant:ident) => {{
            let byte_len = element_count
                .checked_mul(std::mem::size_of::<$native>())
                .ok_or_else(|| pointer_output_error("pointer view byte length overflowed"))?;
            if byte_len > max_bytes {
                return Err(pointer_output_error("pointer view exceeds the copy limit"));
            }
            let bytes = copy_address(pointer, byte_len);
            let values = bytes
                .chunks_exact(std::mem::size_of::<$native>())
                .map(|bytes| <$native>::from_ne_bytes(bytes.try_into().expect("exact chunk")))
                .collect::<Vec<_>>();
            let tensor = runmat_value::Tensor::from_numeric_storage(
                runmat_value::NumericStorage::$variant(values),
                shape.to_vec(),
            )
            .map_err(pointer_output_error)?;
            Ok(Value::Tensor(tensor))
        }};
    }

    match scalar {
        NativeScalar::Bool => {
            let bytes = copy_address(pointer, element_count);
            let logical = runmat_value::LogicalArray::new(
                bytes
                    .into_iter()
                    .map(|value| u8::from(value != 0))
                    .collect(),
                shape.to_vec(),
            )
            .map_err(pointer_output_error)?;
            Ok(Value::LogicalArray(logical))
        }
        NativeScalar::Char | NativeScalar::SignedChar | NativeScalar::I8 => {
            copy_numeric!(i8, I8)
        }
        NativeScalar::UnsignedChar | NativeScalar::U8 => copy_numeric!(u8, U8),
        NativeScalar::Short | NativeScalar::I16 => copy_numeric!(i16, I16),
        NativeScalar::UnsignedShort | NativeScalar::U16 => copy_numeric!(u16, U16),
        NativeScalar::Int | NativeScalar::I32 => copy_numeric!(i32, I32),
        NativeScalar::UnsignedInt | NativeScalar::U32 => copy_numeric!(u32, U32),
        NativeScalar::Long => match std::mem::size_of::<std::ffi::c_long>() {
            4 => copy_numeric!(i32, I32),
            8 => copy_numeric!(i64, I64),
            _ => Err(pointer_output_error("unsupported C long width")),
        },
        NativeScalar::UnsignedLong => match std::mem::size_of::<std::ffi::c_ulong>() {
            4 => copy_numeric!(u32, U32),
            8 => copy_numeric!(u64, U64),
            _ => Err(pointer_output_error("unsupported C unsigned long width")),
        },
        NativeScalar::LongLong | NativeScalar::I64 => copy_numeric!(i64, I64),
        NativeScalar::UnsignedLongLong | NativeScalar::U64 => copy_numeric!(u64, U64),
        NativeScalar::Isize => {
            let byte_len = element_count
                .checked_mul(std::mem::size_of::<isize>())
                .ok_or_else(|| pointer_output_error("pointer view byte length overflowed"))?;
            if byte_len > max_bytes {
                return Err(pointer_output_error("pointer view exceeds the copy limit"));
            }
            let bytes = copy_address(pointer, byte_len);
            let values = bytes
                .chunks_exact(std::mem::size_of::<isize>())
                .map(|bytes| isize::from_ne_bytes(bytes.try_into().expect("exact chunk")) as i64)
                .collect();
            runmat_value::Tensor::from_numeric_storage(
                runmat_value::NumericStorage::I64(values),
                shape.to_vec(),
            )
            .map(Value::Tensor)
            .map_err(pointer_output_error)
        }
        NativeScalar::Usize => {
            let byte_len = element_count
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or_else(|| pointer_output_error("pointer view byte length overflowed"))?;
            if byte_len > max_bytes {
                return Err(pointer_output_error("pointer view exceeds the copy limit"));
            }
            let bytes = copy_address(pointer, byte_len);
            let values = bytes
                .chunks_exact(std::mem::size_of::<usize>())
                .map(|bytes| usize::from_ne_bytes(bytes.try_into().expect("exact chunk")) as u64)
                .collect();
            runmat_value::Tensor::from_numeric_storage(
                runmat_value::NumericStorage::U64(values),
                shape.to_vec(),
            )
            .map(Value::Tensor)
            .map_err(pointer_output_error)
        }
        NativeScalar::F32 => copy_numeric!(f32, F32),
        NativeScalar::F64 => copy_numeric!(f64, F64),
    }
}

fn copy_address(pointer: &NativePointer, byte_len: usize) -> Vec<u8> {
    // SAFETY: `setdatatype` is an explicit caller assertion that the native
    // address is valid for the requested type and dimensions. The caller has
    // already checked a bounded, nonzero byte length. Native execution is
    // isolated by default so an invalid external pointer cannot corrupt the
    // driver process.
    unsafe { std::slice::from_raw_parts(pointer.address().cast::<u8>(), byte_len).to_vec() }
}

fn pointer_output_error(message: impl Into<String>) -> InvocationError {
    InvocationError::Output {
        symbol: "lib.pointer".into(),
        message: message.into(),
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
            PointerBindingTarget::Opaque(pointer) => pointer.address(),
        }
    }
}
