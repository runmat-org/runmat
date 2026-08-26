use std::ffi::c_void;

use libffi::middle::{arg, Arg};
use runmat_value::{IntValue, NumericScalar, NumericStorage, StructValue, Value};

use crate::{NativeLibraryMetadata, NativeScalar, NativeType, Parameter, ParameterDirection};

use super::abi::{scalar_size, type_layout};
use super::InvocationError;

#[derive(Debug, Clone)]
pub(super) enum ScalarSlot {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    Isize(isize),
    Usize(usize),
    F32(f32),
    F64(f64),
}

impl ScalarSlot {
    pub(super) fn ffi_arg(&self) -> Arg<'_> {
        match self {
            Self::I8(value) => arg(value),
            Self::U8(value) => arg(value),
            Self::I16(value) => arg(value),
            Self::U16(value) => arg(value),
            Self::I32(value) => arg(value),
            Self::U32(value) => arg(value),
            Self::I64(value) => arg(value),
            Self::U64(value) => arg(value),
            Self::Isize(value) => arg(value),
            Self::Usize(value) => arg(value),
            Self::F32(value) => arg(value),
            Self::F64(value) => arg(value),
        }
    }

    pub(super) fn address(&mut self) -> *mut c_void {
        match self {
            Self::I8(value) => (value as *mut i8).cast(),
            Self::U8(value) => (value as *mut u8).cast(),
            Self::I16(value) => (value as *mut i16).cast(),
            Self::U16(value) => (value as *mut u16).cast(),
            Self::I32(value) => (value as *mut i32).cast(),
            Self::U32(value) => (value as *mut u32).cast(),
            Self::I64(value) => (value as *mut i64).cast(),
            Self::U64(value) => (value as *mut u64).cast(),
            Self::Isize(value) => (value as *mut isize).cast(),
            Self::Usize(value) => (value as *mut usize).cast(),
            Self::F32(value) => (value as *mut f32).cast(),
            Self::F64(value) => (value as *mut f64).cast(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum PointeeSlot {
    Scalar(ScalarSlot),
    Array {
        storage: NumericStorage,
        shape: Vec<usize>,
    },
    Bytes(Vec<u8>),
    Structure(AlignedStorage),
}

impl PointeeSlot {
    pub(super) fn address(&mut self) -> *mut c_void {
        match self {
            Self::Scalar(value) => value.address(),
            Self::Array { storage, .. } => numeric_storage_address(storage),
            Self::Bytes(bytes) => bytes.as_mut_ptr().cast(),
            Self::Structure(storage) => storage.as_mut_ptr(),
        }
    }
}

#[derive(Debug)]
pub(super) struct PointerSlot {
    pub(super) address: *mut c_void,
    pub(super) pointee: Box<PointeeSlot>,
}

#[derive(Debug)]
pub(super) enum ArgumentSlot {
    Scalar(ScalarSlot),
    Pointer(PointerSlot),
    BoundPointer(*mut c_void),
    Callback(*mut c_void),
    Structure(AlignedStorage),
}

impl ArgumentSlot {
    pub(super) fn ffi_arg(&self) -> Arg<'_> {
        match self {
            Self::Scalar(value) => value.ffi_arg(),
            Self::Pointer(value) => arg(&value.address),
            Self::BoundPointer(value) => arg(value),
            Self::Callback(value) => arg(value),
            Self::Structure(value) => value.ffi_arg(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct AlignedStorage {
    words: Vec<u128>,
    len: usize,
}

impl AlignedStorage {
    pub(super) fn zeroed(len: usize) -> Self {
        let words = len.div_ceil(std::mem::size_of::<u128>());
        Self {
            words: vec![0; words],
            len,
        }
    }

    fn ffi_arg(&self) -> Arg<'_> {
        arg(&self.words[0])
    }

    pub(super) fn as_mut_ptr(&mut self) -> *mut c_void {
        self.words.as_mut_ptr().cast()
    }

    pub(super) fn bytes(&self) -> &[u8] {
        // SAFETY: The byte view stays within the initialized `u128` allocation.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }

    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: The exclusive byte view stays within the initialized
        // allocation and cannot outlive this borrow.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), self.len) }
    }
}

pub(super) fn prepare_arguments(
    symbol: &str,
    parameters: &[Parameter],
    values: &[Value],
    metadata: &NativeLibraryMetadata,
    callbacks: &std::collections::BTreeMap<usize, *mut c_void>,
    pointers: &std::collections::BTreeMap<usize, *mut c_void>,
) -> Result<Vec<ArgumentSlot>, InvocationError> {
    parameters
        .iter()
        .zip(values)
        .enumerate()
        .map(|(index, (parameter, value))| {
            prepare_argument(
                symbol, index, parameter, value, metadata, callbacks, pointers,
            )
        })
        .collect()
}

fn prepare_argument(
    symbol: &str,
    index: usize,
    parameter: &Parameter,
    value: &Value,
    metadata: &NativeLibraryMetadata,
    callbacks: &std::collections::BTreeMap<usize, *mut c_void>,
    pointers: &std::collections::BTreeMap<usize, *mut c_void>,
) -> Result<ArgumentSlot, InvocationError> {
    if let Some(pointer) = pointers.get(&index) {
        if !matches!(parameter.ty, NativeType::Pointer { .. }) {
            return Err(InvocationError::Argument {
                symbol: symbol.into(),
                argument: index + 1,
                name: parameter.name.clone(),
                message: "pointer binding points to a non-pointer argument".into(),
            });
        }
        return Ok(ArgumentSlot::BoundPointer(*pointer));
    }
    let result = match &parameter.ty {
        NativeType::Scalar { scalar } => {
            scalar_from_value(*scalar, value).map(ArgumentSlot::Scalar)
        }
        NativeType::Enumeration { storage, .. } => {
            scalar_from_value(*storage, value).map(ArgumentSlot::Scalar)
        }
        NativeType::Pointer { pointee, .. } => pointee_from_value(symbol, pointee, value, metadata)
            .map(|pointee| {
                let mut pointee = Box::new(pointee);
                let address = pointee.address();
                ArgumentSlot::Pointer(PointerSlot { address, pointee })
            }),
        NativeType::Structure { .. } | NativeType::Array { .. } => {
            pack_value(symbol, &parameter.ty, value, metadata).map(ArgumentSlot::Structure)
        }
        NativeType::Callback { .. } => callbacks
            .get(&index)
            .copied()
            .map(ArgumentSlot::Callback)
            .ok_or_else(|| "callback argument requires a callback binding".into()),
        NativeType::Void => Err(format!(
            "{} cannot be used as a direct argument",
            type_name(&parameter.ty)
        )),
    };
    result.map_err(|message| InvocationError::Argument {
        symbol: symbol.into(),
        argument: index + 1,
        name: parameter.name.clone(),
        message,
    })
}

pub(super) fn pointee_from_value(
    symbol: &str,
    pointee: &NativeType,
    value: &Value,
    metadata: &NativeLibraryMetadata,
) -> Result<PointeeSlot, String> {
    match pointee {
        NativeType::Scalar { scalar }
        | NativeType::Enumeration {
            storage: scalar, ..
        } => {
            if let Value::Tensor(tensor) = value {
                let storage = tensor
                    .clone()
                    .into_numeric_storage()
                    .map_err(|message| format!("could not access numeric array: {message}"))?;
                ensure_storage_matches(*scalar, &storage)?;
                Ok(PointeeSlot::Array {
                    storage,
                    shape: tensor.shape.clone(),
                })
            } else if matches!(
                scalar,
                NativeScalar::Char | NativeScalar::SignedChar | NativeScalar::I8
            ) && matches!(value, Value::String(_) | Value::CharArray(_))
            {
                let text = String::try_from(value).map_err(|message| message.to_string())?;
                if text.as_bytes().contains(&0) {
                    return Err("string contains an interior NUL byte".into());
                }
                let mut bytes = text.into_bytes();
                bytes.push(0);
                Ok(PointeeSlot::Bytes(bytes))
            } else {
                scalar_from_value(*scalar, value).map(PointeeSlot::Scalar)
            }
        }
        NativeType::Structure { .. } | NativeType::Array { .. } => {
            pack_value(symbol, pointee, value, metadata).map(PointeeSlot::Structure)
        }
        NativeType::Void => Err("void pointers require a typed libpointer value".into()),
        NativeType::Pointer { .. } => Err("pointer-to-pointer arguments require libpointer".into()),
        NativeType::Callback { .. } => Err("callback pointers require a function handle".into()),
    }
}

pub(super) fn scalar_from_value(scalar: NativeScalar, value: &Value) -> Result<ScalarSlot, String> {
    let number = numeric_scalar(value)?;
    macro_rules! signed {
        ($variant:ident, $type:ty) => {{
            let value = exact_i128(number)?;
            <$type>::try_from(value)
                .map(ScalarSlot::$variant)
                .map_err(|_| {
                    format!(
                        "value {value} is outside the range of {}",
                        stringify!($type)
                    )
                })
        }};
    }
    macro_rules! unsigned {
        ($variant:ident, $type:ty) => {{
            let value = exact_u128(number)?;
            <$type>::try_from(value)
                .map(ScalarSlot::$variant)
                .map_err(|_| {
                    format!(
                        "value {value} is outside the range of {}",
                        stringify!($type)
                    )
                })
        }};
    }
    match scalar {
        NativeScalar::Bool => Ok(ScalarSlot::U8(u8::from(number.materialize_f64() != 0.0))),
        NativeScalar::Char => signed!(I8, std::ffi::c_char),
        NativeScalar::SignedChar | NativeScalar::I8 => signed!(I8, i8),
        NativeScalar::UnsignedChar | NativeScalar::U8 => unsigned!(U8, u8),
        NativeScalar::Short => signed!(I16, std::ffi::c_short),
        NativeScalar::UnsignedShort => unsigned!(U16, std::ffi::c_ushort),
        NativeScalar::Int => signed!(I32, std::ffi::c_int),
        NativeScalar::UnsignedInt => unsigned!(U32, std::ffi::c_uint),
        NativeScalar::Long => match std::mem::size_of::<std::ffi::c_long>() {
            4 => signed!(I32, i32),
            8 => signed!(I64, i64),
            _ => Err("unsupported C long width".into()),
        },
        NativeScalar::UnsignedLong => match std::mem::size_of::<std::ffi::c_ulong>() {
            4 => unsigned!(U32, u32),
            8 => unsigned!(U64, u64),
            _ => Err("unsupported C unsigned long width".into()),
        },
        NativeScalar::LongLong | NativeScalar::I64 => signed!(I64, i64),
        NativeScalar::UnsignedLongLong | NativeScalar::U64 => unsigned!(U64, u64),
        NativeScalar::I16 => signed!(I16, i16),
        NativeScalar::U16 => unsigned!(U16, u16),
        NativeScalar::I32 => signed!(I32, i32),
        NativeScalar::U32 => unsigned!(U32, u32),
        NativeScalar::Isize => signed!(Isize, isize),
        NativeScalar::Usize => unsigned!(Usize, usize),
        NativeScalar::F32 => Ok(ScalarSlot::F32(number.materialize_f64() as f32)),
        NativeScalar::F64 => Ok(ScalarSlot::F64(number.materialize_f64())),
    }
}

fn numeric_scalar(value: &Value) -> Result<NumericScalar, String> {
    match value {
        Value::Num(value) => Ok(NumericScalar::F64(*value)),
        Value::Int(value) => Ok(NumericScalar::from(value.clone())),
        Value::Bool(value) => Ok(NumericScalar::U8(u8::from(*value))),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .ok_or_else(|| "empty array cannot be converted to a scalar".into()),
        _ => Err("expected a real numeric scalar".into()),
    }
}

fn exact_i128(value: NumericScalar) -> Result<i128, String> {
    match value {
        NumericScalar::F64(value) => exact_float_i128(value),
        NumericScalar::F32(value) => exact_float_i128(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_i64())
            .map(i128::from)
            .ok_or_else(|| "unsigned integer exceeds the signed range".into()),
    }
}

fn exact_u128(value: NumericScalar) -> Result<u128, String> {
    match value {
        NumericScalar::F64(value) => exact_float_u128(value),
        NumericScalar::F32(value) => exact_float_u128(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_u64())
            .map(u128::from)
            .ok_or_else(|| "negative integer cannot be converted to an unsigned type".into()),
    }
}

fn exact_float_i128(value: f64) -> Result<i128, String> {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= i128::MIN as f64
        && value <= i128::MAX as f64
    {
        Ok(value as i128)
    } else {
        Err(format!(
            "floating value {value} is not an exactly representable integer"
        ))
    }
}

fn exact_float_u128(value: f64) -> Result<u128, String> {
    if value.is_finite() && value.fract() == 0.0 && value >= 0.0 && value <= u128::MAX as f64 {
        Ok(value as u128)
    } else {
        Err(format!(
            "floating value {value} is not an exactly representable unsigned integer"
        ))
    }
}

fn ensure_storage_matches(scalar: NativeScalar, storage: &NumericStorage) -> Result<(), String> {
    let matches = matches!(
        (scalar, storage),
        (NativeScalar::F64, NumericStorage::F64(_))
            | (NativeScalar::F32, NumericStorage::F32(_))
            | (
                NativeScalar::I8 | NativeScalar::SignedChar | NativeScalar::Char,
                NumericStorage::I8(_)
            )
            | (
                NativeScalar::U8 | NativeScalar::UnsignedChar | NativeScalar::Bool,
                NumericStorage::U8(_)
            )
            | (
                NativeScalar::I16 | NativeScalar::Short,
                NumericStorage::I16(_)
            )
            | (
                NativeScalar::U16 | NativeScalar::UnsignedShort,
                NumericStorage::U16(_)
            )
            | (
                NativeScalar::I32 | NativeScalar::Int,
                NumericStorage::I32(_)
            )
            | (
                NativeScalar::U32 | NativeScalar::UnsignedInt,
                NumericStorage::U32(_)
            )
            | (
                NativeScalar::I64 | NativeScalar::LongLong,
                NumericStorage::I64(_)
            )
            | (
                NativeScalar::U64 | NativeScalar::UnsignedLongLong,
                NumericStorage::U64(_)
            )
    );
    if matches {
        Ok(())
    } else {
        Err(format!(
            "array class {} does not match pointer element type {scalar:?}",
            storage.class_name()
        ))
    }
}

fn numeric_storage_address(storage: &mut NumericStorage) -> *mut c_void {
    match storage {
        NumericStorage::F64(values) => values.as_mut_ptr().cast(),
        NumericStorage::F32(values) => values.as_mut_ptr().cast(),
        NumericStorage::I8(values) => values.as_mut_ptr().cast(),
        NumericStorage::I16(values) => values.as_mut_ptr().cast(),
        NumericStorage::I32(values) => values.as_mut_ptr().cast(),
        NumericStorage::I64(values) => values.as_mut_ptr().cast(),
        NumericStorage::U8(values) => values.as_mut_ptr().cast(),
        NumericStorage::U16(values) => values.as_mut_ptr().cast(),
        NumericStorage::U32(values) => values.as_mut_ptr().cast(),
        NumericStorage::U64(values) => values.as_mut_ptr().cast(),
    }
}

pub(super) fn pack_value(
    symbol: &str,
    ty: &NativeType,
    value: &Value,
    metadata: &NativeLibraryMetadata,
) -> Result<AlignedStorage, String> {
    let (size, _) = type_layout(symbol, ty, metadata).map_err(|error| error.to_string())?;
    let mut storage = AlignedStorage::zeroed(size);
    pack_into(symbol, ty, value, metadata, storage.bytes_mut())?;
    Ok(storage)
}

fn pack_into(
    symbol: &str,
    ty: &NativeType,
    value: &Value,
    metadata: &NativeLibraryMetadata,
    destination: &mut [u8],
) -> Result<(), String> {
    match ty {
        NativeType::Scalar { scalar }
        | NativeType::Enumeration {
            storage: scalar, ..
        } => {
            let slot = scalar_from_value(*scalar, value)?;
            write_scalar(&slot, destination)
        }
        NativeType::Structure { name } => {
            let Value::Struct(value) = value else {
                return Err(format!("expected a structure value for `{name}`"));
            };
            pack_structure(symbol, name, value, metadata, destination)
        }
        _ => Err(format!(
            "packing {} by value is not implemented",
            type_name(ty)
        )),
    }
}

fn pack_structure(
    symbol: &str,
    name: &str,
    value: &StructValue,
    metadata: &NativeLibraryMetadata,
    destination: &mut [u8],
) -> Result<(), String> {
    let definition = metadata
        .structures
        .iter()
        .find(|definition| definition.name == name)
        .ok_or_else(|| format!("missing structure definition `{name}`"))?;
    let mut type_ = super::abi::ffi_type(
        symbol,
        &NativeType::Structure { name: name.into() },
        metadata,
    )
    .map_err(|error| error.to_string())?;
    let offsets = type_
        .struct_offsets(libffi::middle::ffi_abi_FFI_DEFAULT_ABI)
        .map_err(|error| format!("could not lay out structure `{name}`: {error:?}"))?;
    for (field, offset) in definition.fields.iter().zip(offsets) {
        let field_value = value
            .fields
            .get(&field.name)
            .ok_or_else(|| format!("structure `{name}` is missing field `{}`", field.name))?;
        let (field_size, _) =
            type_layout(symbol, &field.ty, metadata).map_err(|error| error.to_string())?;
        let end = offset
            .checked_add(field_size)
            .ok_or_else(|| format!("field `{}` layout overflows", field.name))?;
        let field_destination = destination
            .get_mut(offset..end)
            .ok_or_else(|| format!("field `{}` exceeds structure storage", field.name))?;
        pack_into(symbol, &field.ty, field_value, metadata, field_destination)?;
    }
    Ok(())
}

fn write_scalar(slot: &ScalarSlot, destination: &mut [u8]) -> Result<(), String> {
    macro_rules! write {
        ($value:expr) => {{
            let bytes = $value.to_ne_bytes();
            destination
                .get_mut(..bytes.len())
                .ok_or_else(|| "scalar destination is too small".to_string())?
                .copy_from_slice(&bytes);
            Ok(())
        }};
    }
    match slot {
        ScalarSlot::I8(value) => destination
            .first_mut()
            .map(|slot| *slot = *value as u8)
            .ok_or_else(|| "scalar destination is empty".into()),
        ScalarSlot::U8(value) => destination
            .first_mut()
            .map(|slot| *slot = *value)
            .ok_or_else(|| "scalar destination is empty".into()),
        ScalarSlot::I16(value) => write!(value),
        ScalarSlot::U16(value) => write!(value),
        ScalarSlot::I32(value) => write!(value),
        ScalarSlot::U32(value) => write!(value),
        ScalarSlot::I64(value) => write!(value),
        ScalarSlot::U64(value) => write!(value),
        ScalarSlot::Isize(value) => write!(value),
        ScalarSlot::Usize(value) => write!(value),
        ScalarSlot::F32(value) => write!(value),
        ScalarSlot::F64(value) => write!(value),
    }
}

pub(super) fn type_name(ty: &NativeType) -> &'static str {
    match ty {
        NativeType::Void => "void",
        NativeType::Scalar { .. } => "scalar",
        NativeType::Pointer { .. } => "pointer",
        NativeType::Array { .. } => "array",
        NativeType::Structure { .. } => "structure",
        NativeType::Enumeration { .. } => "enumeration",
        NativeType::Callback { .. } => "callback",
    }
}

pub(super) fn is_output(parameter: &Parameter) -> bool {
    !matches!(parameter.direction, ParameterDirection::Input)
}

pub(super) fn pointee_value(slot: &PointeeSlot, scalar: NativeScalar) -> Result<Value, String> {
    match slot {
        PointeeSlot::Scalar(value) => Ok(scalar_slot_value(scalar, value)),
        PointeeSlot::Array { storage, shape } => {
            runmat_value::Tensor::from_numeric_storage(storage.clone(), shape.clone())
                .map(Value::Tensor)
        }
        PointeeSlot::Bytes(bytes) => {
            let end = bytes
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(bytes.len());
            String::from_utf8(bytes[..end].to_vec())
                .map(Value::String)
                .map_err(|error| format!("native string is not UTF-8: {error}"))
        }
        PointeeSlot::Structure(_) => Err("structured output decoding is handled separately".into()),
    }
}

pub(super) fn scalar_slot_value(scalar: NativeScalar, slot: &ScalarSlot) -> Value {
    match (scalar, slot) {
        (NativeScalar::Bool, ScalarSlot::U8(value)) => Value::Bool(*value != 0),
        (NativeScalar::F32, ScalarSlot::F32(value)) => Value::Num(f64::from(*value)),
        (NativeScalar::F64, ScalarSlot::F64(value)) => Value::Num(*value),
        (_, ScalarSlot::I8(value)) => Value::Int(IntValue::I8(*value)),
        (_, ScalarSlot::U8(value)) => Value::Int(IntValue::U8(*value)),
        (_, ScalarSlot::I16(value)) => Value::Int(IntValue::I16(*value)),
        (_, ScalarSlot::U16(value)) => Value::Int(IntValue::U16(*value)),
        (_, ScalarSlot::I32(value)) => Value::Int(IntValue::I32(*value)),
        (_, ScalarSlot::U32(value)) => Value::Int(IntValue::U32(*value)),
        (_, ScalarSlot::I64(value)) => Value::Int(IntValue::I64(*value)),
        (_, ScalarSlot::U64(value)) => Value::Int(IntValue::U64(*value)),
        (_, ScalarSlot::Isize(value)) => Value::Int(IntValue::I64(*value as i64)),
        (_, ScalarSlot::Usize(value)) => Value::Int(IntValue::U64(*value as u64)),
        (_, ScalarSlot::F32(value)) => Value::Num(f64::from(*value)),
        (_, ScalarSlot::F64(value)) => Value::Num(*value),
    }
}

pub(super) fn zero_scalar(scalar: NativeScalar) -> ScalarSlot {
    match scalar_size(scalar) {
        1 if matches!(
            scalar,
            NativeScalar::Char | NativeScalar::SignedChar | NativeScalar::I8
        ) =>
        {
            ScalarSlot::I8(0)
        }
        1 => ScalarSlot::U8(0),
        2 if matches!(scalar, NativeScalar::Short | NativeScalar::I16) => ScalarSlot::I16(0),
        2 => ScalarSlot::U16(0),
        4 if matches!(scalar, NativeScalar::F32) => ScalarSlot::F32(0.0),
        4 if matches!(
            scalar,
            NativeScalar::Int | NativeScalar::I32 | NativeScalar::Long
        ) =>
        {
            ScalarSlot::I32(0)
        }
        4 => ScalarSlot::U32(0),
        8 if matches!(scalar, NativeScalar::F64) => ScalarSlot::F64(0.0),
        8 if matches!(
            scalar,
            NativeScalar::Long | NativeScalar::LongLong | NativeScalar::I64 | NativeScalar::Isize
        ) =>
        {
            if matches!(scalar, NativeScalar::Isize) {
                ScalarSlot::Isize(0)
            } else {
                ScalarSlot::I64(0)
            }
        }
        8 if matches!(scalar, NativeScalar::Usize) => ScalarSlot::Usize(0),
        8 => ScalarSlot::U64(0),
        _ => ScalarSlot::U64(0),
    }
}

pub(super) unsafe fn write_scalar_to_pointer(slot: &ScalarSlot, destination: *mut c_void) {
    match slot {
        ScalarSlot::I8(value) => unsafe { destination.cast::<i8>().write(*value) },
        ScalarSlot::U8(value) => unsafe { destination.cast::<u8>().write(*value) },
        ScalarSlot::I16(value) => unsafe { destination.cast::<i16>().write(*value) },
        ScalarSlot::U16(value) => unsafe { destination.cast::<u16>().write(*value) },
        ScalarSlot::I32(value) => unsafe { destination.cast::<i32>().write(*value) },
        ScalarSlot::U32(value) => unsafe { destination.cast::<u32>().write(*value) },
        ScalarSlot::I64(value) => unsafe { destination.cast::<i64>().write(*value) },
        ScalarSlot::U64(value) => unsafe { destination.cast::<u64>().write(*value) },
        ScalarSlot::Isize(value) => unsafe { destination.cast::<isize>().write(*value) },
        ScalarSlot::Usize(value) => unsafe { destination.cast::<usize>().write(*value) },
        ScalarSlot::F32(value) => unsafe { destination.cast::<f32>().write(*value) },
        ScalarSlot::F64(value) => unsafe { destination.cast::<f64>().write(*value) },
    }
}
