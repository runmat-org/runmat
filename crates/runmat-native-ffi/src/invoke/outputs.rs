use runmat_value::Value;

use crate::{
    NativeLibraryMetadata, NativePointer, NativeScalar, NativeType, Parameter, SymbolPrototype,
};

use super::arguments::{
    is_output, pointee_value, scalar_slot_value, ArgumentSlot, PointeeSlot, ScalarSlot,
};
use super::{InvocationError, InvocationValue};

#[derive(Debug)]
pub(super) enum ReturnSlot {
    Void,
    Scalar(ScalarSlot),
    Pointer(*mut std::ffi::c_void),
    Structure(super::arguments::AlignedStorage),
}

pub(super) fn return_slot(
    symbol: &str,
    ty: &NativeType,
    metadata: &NativeLibraryMetadata,
) -> Result<ReturnSlot, InvocationError> {
    match ty {
        NativeType::Void => Ok(ReturnSlot::Void),
        NativeType::Scalar { scalar }
        | NativeType::Enumeration {
            storage: scalar, ..
        } => Ok(ReturnSlot::Scalar(super::arguments::zero_scalar(*scalar))),
        NativeType::Pointer { .. } | NativeType::Callback { .. } => {
            Ok(ReturnSlot::Pointer(std::ptr::null_mut()))
        }
        NativeType::Structure { .. } | NativeType::Array { .. } => {
            let (size, _) = super::abi::type_layout(symbol, ty, metadata)?;
            Ok(ReturnSlot::Structure(
                super::arguments::AlignedStorage::zeroed(size),
            ))
        }
    }
}

pub(super) fn decode_return(
    symbol: &str,
    prototype: &SymbolPrototype,
    slot: &ReturnSlot,
    metadata: &NativeLibraryMetadata,
) -> Result<Option<InvocationValue>, InvocationError> {
    let ty = &prototype.return_type;
    let result = match (ty, slot) {
        (NativeType::Void, ReturnSlot::Void) => None,
        (NativeType::Scalar { scalar }, ReturnSlot::Scalar(value))
        | (
            NativeType::Enumeration {
                storage: scalar, ..
            },
            ReturnSlot::Scalar(value),
        ) => Some(InvocationValue::Value(scalar_slot_value(*scalar, value))),
        (NativeType::Pointer { pointee, .. }, ReturnSlot::Pointer(pointer)) => {
            match std::ptr::NonNull::new(*pointer) {
                Some(address) => {
                    // SAFETY: The function returned this address under the
                    // ownership and pointee contract in the validated prototype.
                    let pointer = unsafe {
                        NativePointer::from_address(
                            address,
                            pointee.as_ref().clone(),
                            prototype
                                .return_ownership
                                .ok_or_else(|| InvocationError::Output {
                                    symbol: symbol.into(),
                                    message: "pointer return has no ownership contract".into(),
                                })?,
                        )
                    };
                    Some(InvocationValue::Pointer(pointer))
                }
                None if prototype.return_nullable => None,
                None => {
                    return Err(InvocationError::Output {
                        symbol: symbol.into(),
                        message: "non-null pointer return was null".into(),
                    })
                }
            }
        }
        (NativeType::Structure { .. }, ReturnSlot::Structure(storage)) => Some(
            InvocationValue::Value(decode_structure(symbol, ty, storage.bytes(), metadata)?),
        ),
        _ => {
            return Err(InvocationError::Output {
                symbol: symbol.into(),
                message: "return storage does not match the normalized prototype".into(),
            })
        }
    };
    Ok(result)
}

pub(super) fn decode_output_parameters(
    symbol: &str,
    prototype: &SymbolPrototype,
    slots: &[ArgumentSlot],
    metadata: &NativeLibraryMetadata,
) -> Result<Vec<(usize, Value)>, InvocationError> {
    prototype
        .parameters
        .iter()
        .zip(slots)
        .enumerate()
        .filter(|(_, (parameter, _))| is_output(parameter))
        .map(|(index, (parameter, slot))| {
            let value =
                decode_output_parameter(symbol, parameter, slot, metadata).map_err(|message| {
                    InvocationError::Output {
                        symbol: symbol.into(),
                        message: format!("parameter {} ({}): {message}", index + 1, parameter.name),
                    }
                })?;
            Ok((index, value))
        })
        .collect()
}

fn decode_output_parameter(
    symbol: &str,
    parameter: &Parameter,
    slot: &ArgumentSlot,
    metadata: &NativeLibraryMetadata,
) -> Result<Value, String> {
    let NativeType::Pointer { pointee, .. } = &parameter.ty else {
        return Err("output parameter is not a pointer".into());
    };
    let ArgumentSlot::Pointer(pointer) = slot else {
        return Err("output argument storage is not a pointer".into());
    };
    match pointee.as_ref() {
        NativeType::Scalar { scalar }
        | NativeType::Enumeration {
            storage: scalar, ..
        } => pointee_value(pointer, *scalar),
        NativeType::Structure { .. } => {
            let PointeeSlot::Structure(storage) = &*pointer.pointee else {
                return Err("structured output storage is invalid".into());
            };
            decode_structure(symbol, pointee, storage.bytes(), metadata)
                .map_err(|error| error.to_string())
        }
        _ => Err("output pointee type is not yet supported".into()),
    }
}

fn decode_structure(
    symbol: &str,
    ty: &NativeType,
    bytes: &[u8],
    metadata: &NativeLibraryMetadata,
) -> Result<Value, InvocationError> {
    let NativeType::Structure { name } = ty else {
        return Err(InvocationError::Output {
            symbol: symbol.into(),
            message: "expected a structure type".into(),
        });
    };
    let definition = metadata
        .structures
        .iter()
        .find(|definition| definition.name == *name)
        .ok_or_else(|| InvocationError::Output {
            symbol: symbol.into(),
            message: format!("missing structure definition `{name}`"),
        })?;
    let mut ffi_type = super::abi::ffi_type(symbol, ty, metadata)?;
    let offsets = ffi_type
        .struct_offsets(libffi::middle::ffi_abi_FFI_DEFAULT_ABI)
        .map_err(|error| InvocationError::Output {
            symbol: symbol.into(),
            message: format!("could not lay out structure `{name}`: {error:?}"),
        })?;
    let mut value = runmat_value::StructValue::new();
    for (field, offset) in definition.fields.iter().zip(offsets) {
        let field_value = decode_field(symbol, &field.ty, &bytes[offset..], metadata)?;
        value.insert(field.name.clone(), field_value);
    }
    Ok(Value::Struct(value))
}

fn decode_field(
    symbol: &str,
    ty: &NativeType,
    bytes: &[u8],
    metadata: &NativeLibraryMetadata,
) -> Result<Value, InvocationError> {
    match ty {
        NativeType::Scalar { scalar }
        | NativeType::Enumeration {
            storage: scalar, ..
        } => {
            let slot = read_scalar(*scalar, bytes).map_err(|message| InvocationError::Output {
                symbol: symbol.into(),
                message,
            })?;
            Ok(scalar_slot_value(*scalar, &slot))
        }
        NativeType::Structure { .. } => decode_structure(symbol, ty, bytes, metadata),
        _ => Err(InvocationError::Output {
            symbol: symbol.into(),
            message: "pointer and array structure fields require explicit ownership metadata"
                .into(),
        }),
    }
}

fn read_scalar(scalar: NativeScalar, bytes: &[u8]) -> Result<ScalarSlot, String> {
    macro_rules! read {
        ($variant:ident, $type:ty) => {{
            let width = std::mem::size_of::<$type>();
            let source: [u8; std::mem::size_of::<$type>()] = bytes
                .get(..width)
                .ok_or_else(|| "native scalar storage is truncated".to_string())?
                .try_into()
                .map_err(|_| "native scalar storage has an invalid width".to_string())?;
            ScalarSlot::$variant(<$type>::from_ne_bytes(source))
        }};
    }
    Ok(match scalar {
        NativeScalar::Bool | NativeScalar::U8 | NativeScalar::UnsignedChar => read!(U8, u8),
        NativeScalar::Char | NativeScalar::SignedChar | NativeScalar::I8 => read!(I8, i8),
        NativeScalar::Short | NativeScalar::I16 => read!(I16, i16),
        NativeScalar::UnsignedShort | NativeScalar::U16 => read!(U16, u16),
        NativeScalar::Int | NativeScalar::I32 => read!(I32, i32),
        NativeScalar::UnsignedInt | NativeScalar::U32 => read!(U32, u32),
        NativeScalar::Long => match std::mem::size_of::<std::ffi::c_long>() {
            4 => read!(I32, i32),
            8 => read!(I64, i64),
            _ => return Err("unsupported C long width".into()),
        },
        NativeScalar::UnsignedLong => match std::mem::size_of::<std::ffi::c_ulong>() {
            4 => read!(U32, u32),
            8 => read!(U64, u64),
            _ => return Err("unsupported C unsigned long width".into()),
        },
        NativeScalar::LongLong | NativeScalar::I64 => read!(I64, i64),
        NativeScalar::UnsignedLongLong | NativeScalar::U64 => read!(U64, u64),
        NativeScalar::Isize => read!(Isize, isize),
        NativeScalar::Usize => read!(Usize, usize),
        NativeScalar::F32 => read!(F32, f32),
        NativeScalar::F64 => read!(F64, f64),
    })
}
