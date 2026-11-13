use std::mem;

use libffi::middle::{ffi_abi_FFI_DEFAULT_ABI, FfiAbi, Type};

use crate::{
    CallingConvention, NativeLibraryMetadata, NativeScalar, NativeType, StructureDefinition,
};

use super::InvocationError;

pub(super) fn calling_convention(
    symbol: &str,
    convention: CallingConvention,
) -> Result<FfiAbi, InvocationError> {
    match convention {
        CallingConvention::C | CallingConvention::System => Ok(ffi_abi_FFI_DEFAULT_ABI),
        CallingConvention::Stdcall
        | CallingConvention::Fastcall
        | CallingConvention::Thiscall
        | CallingConvention::Vectorcall => Err(InvocationError::Abi {
            symbol: symbol.into(),
            message: format!(
                "calling convention {convention:?} is not available on this target through the portable ABI path"
            ),
        }),
    }
}

pub(super) fn ffi_type(
    symbol: &str,
    ty: &NativeType,
    metadata: &NativeLibraryMetadata,
) -> Result<Type, InvocationError> {
    match ty {
        NativeType::Void => Ok(Type::void()),
        NativeType::Scalar { scalar } => Ok(scalar_type(*scalar)),
        NativeType::Pointer { .. } | NativeType::Callback { .. } => Ok(Type::pointer()),
        NativeType::Array { element, length } => {
            let fields = (0..*length)
                .map(|_| ffi_type(symbol, element, metadata))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Type::structure(fields))
        }
        NativeType::Structure { name } => {
            let definition = metadata
                .structures
                .iter()
                .find(|definition| definition.name == *name)
                .ok_or_else(|| InvocationError::Abi {
                    symbol: symbol.into(),
                    message: format!("missing structure definition `{name}`"),
                })?;
            structure_type(symbol, definition, metadata)
        }
        NativeType::Enumeration { storage, .. } => Ok(scalar_type(*storage)),
    }
}

fn structure_type(
    symbol: &str,
    definition: &StructureDefinition,
    metadata: &NativeLibraryMetadata,
) -> Result<Type, InvocationError> {
    definition
        .fields
        .iter()
        .map(|field| ffi_type(symbol, &field.ty, metadata))
        .collect::<Result<Vec<_>, _>>()
        .map(Type::structure)
}

pub(super) fn scalar_type(scalar: NativeScalar) -> Type {
    match scalar {
        NativeScalar::Bool | NativeScalar::U8 => Type::u8(),
        NativeScalar::Char => Type::c_schar(),
        NativeScalar::SignedChar | NativeScalar::I8 => Type::i8(),
        NativeScalar::UnsignedChar => Type::c_uchar(),
        NativeScalar::Short => Type::c_short(),
        NativeScalar::UnsignedShort => Type::c_ushort(),
        NativeScalar::Int => Type::c_int(),
        NativeScalar::UnsignedInt => Type::c_uint(),
        NativeScalar::Long => Type::c_long(),
        NativeScalar::UnsignedLong => Type::c_ulong(),
        NativeScalar::LongLong => Type::c_longlong(),
        NativeScalar::UnsignedLongLong => Type::c_ulonglong(),
        NativeScalar::I16 => Type::i16(),
        NativeScalar::U16 => Type::u16(),
        NativeScalar::I32 => Type::i32(),
        NativeScalar::U32 => Type::u32(),
        NativeScalar::I64 => Type::i64(),
        NativeScalar::U64 => Type::u64(),
        NativeScalar::Isize => Type::isize(),
        NativeScalar::Usize => Type::usize(),
        NativeScalar::F32 => Type::f32(),
        NativeScalar::F64 => Type::f64(),
    }
}

pub(super) fn type_layout(
    symbol: &str,
    ty: &NativeType,
    metadata: &NativeLibraryMetadata,
) -> Result<(usize, usize), InvocationError> {
    let mut type_ = ffi_type(symbol, ty, metadata)?;
    if matches!(ty, NativeType::Structure { .. } | NativeType::Array { .. }) {
        type_
            .struct_offsets(ffi_abi_FFI_DEFAULT_ABI)
            .map_err(|error| InvocationError::Abi {
                symbol: symbol.into(),
                message: format!("could not prepare type layout: {error:?}"),
            })?;
    }
    // SAFETY: `Type` owns a valid libffi type descriptor for the duration of
    // these reads, and structured types were laid out immediately above.
    let raw = type_.as_raw_ptr();
    // SAFETY: `raw` is non-null and owned by `type_` above.
    let (size, alignment) = unsafe { ((*raw).size, usize::from((*raw).alignment)) };
    if size == 0 && !matches!(ty, NativeType::Void) {
        return Err(InvocationError::Abi {
            symbol: symbol.into(),
            message: "libffi produced a zero-sized non-void type".into(),
        });
    }
    Ok((size, alignment.max(1)))
}

pub(super) fn scalar_size(scalar: NativeScalar) -> usize {
    match scalar {
        NativeScalar::Bool | NativeScalar::U8 => 1,
        NativeScalar::Char => mem::size_of::<std::ffi::c_char>(),
        NativeScalar::SignedChar | NativeScalar::I8 => 1,
        NativeScalar::UnsignedChar => mem::size_of::<std::ffi::c_uchar>(),
        NativeScalar::Short => mem::size_of::<std::ffi::c_short>(),
        NativeScalar::UnsignedShort => mem::size_of::<std::ffi::c_ushort>(),
        NativeScalar::Int => mem::size_of::<std::ffi::c_int>(),
        NativeScalar::UnsignedInt => mem::size_of::<std::ffi::c_uint>(),
        NativeScalar::Long => mem::size_of::<std::ffi::c_long>(),
        NativeScalar::UnsignedLong => mem::size_of::<std::ffi::c_ulong>(),
        NativeScalar::LongLong => mem::size_of::<std::ffi::c_longlong>(),
        NativeScalar::UnsignedLongLong => mem::size_of::<std::ffi::c_ulonglong>(),
        NativeScalar::I16 | NativeScalar::U16 => 2,
        NativeScalar::I32 | NativeScalar::U32 | NativeScalar::F32 => 4,
        NativeScalar::I64 | NativeScalar::U64 | NativeScalar::F64 => 8,
        NativeScalar::Isize | NativeScalar::Usize => mem::size_of::<usize>(),
    }
}
