//! Exact scalar attribute values for simple HDF5 display.

use super::*;
use std::fmt::Display;

pub(super) fn scalar_attribute_text(
    builtin: &'static str,
    attr: &Attribute,
    dtype: &TypeDescriptor,
) -> BuiltinResult<Option<String>> {
    if !attr.shape().iter().all(|extent| *extent == 1) {
        return Ok(None);
    }
    let value = match dtype {
        TypeDescriptor::Integer(IntSize::U1) => read_display::<i8>(builtin, attr)?,
        TypeDescriptor::Integer(IntSize::U2) => read_display::<i16>(builtin, attr)?,
        TypeDescriptor::Integer(IntSize::U4) => read_display::<i32>(builtin, attr)?,
        TypeDescriptor::Integer(IntSize::U8) => read_display::<i64>(builtin, attr)?,
        TypeDescriptor::Unsigned(IntSize::U1) => read_display::<u8>(builtin, attr)?,
        TypeDescriptor::Unsigned(IntSize::U2) => read_display::<u16>(builtin, attr)?,
        TypeDescriptor::Unsigned(IntSize::U4) => read_display::<u32>(builtin, attr)?,
        TypeDescriptor::Unsigned(IntSize::U8) => read_display::<u64>(builtin, attr)?,
        TypeDescriptor::Float(FloatSize::U4) => read_display::<f32>(builtin, attr)?,
        TypeDescriptor::Float(FloatSize::U8) => read_display::<f64>(builtin, attr)?,
        TypeDescriptor::Boolean => read_display::<bool>(builtin, attr)?,
        TypeDescriptor::VarLenUnicode => {
            read_one::<VarLenUnicode>(builtin, attr)?.map(|value| value.as_str().to_string())
        }
        TypeDescriptor::VarLenAscii => {
            read_one::<VarLenAscii>(builtin, attr)?.map(|value| value.as_str().to_string())
        }
        _ => None,
    };
    Ok(value)
}

fn read_display<T>(builtin: &'static str, attr: &Attribute) -> BuiltinResult<Option<String>>
where
    T: hdf5::H5Type + Display,
{
    Ok(read_one::<T>(builtin, attr)?.map(|value| value.to_string()))
}

fn read_one<T>(builtin: &'static str, attr: &Attribute) -> BuiltinResult<Option<T>>
where
    T: hdf5::H5Type,
{
    let values = attr.read_raw::<T>().map_err(|source| {
        io_error(
            builtin,
            format!("{builtin}: unable to read scalar attribute value"),
            source,
        )
    })?;
    Ok(values.into_iter().next())
}
