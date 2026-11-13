//! Native scalar adaptation only. Conversion policy remains with the canonical
//! integer assignment owner and the existing checked fread char boundary.

use super::*;
use runmat_value::Utf16CodeUnit;

pub(super) trait ReadValue: Copy {
    fn numeric(self) -> NumericScalar;
    fn character(self) -> Result<Utf16CodeUnit, String> {
        numeric_character(self.numeric())
    }
}

impl ReadValue for Utf16CodeUnit {
    fn numeric(self) -> NumericScalar {
        NumericScalar::U16(self.raw())
    }
    fn character(self) -> Result<Utf16CodeUnit, String> {
        Ok(self)
    }
}

pub(super) trait FixedValue: ReadValue {
    const INPUT: InputType;
    fn decode(bytes: &[u8], endianness: Endianness) -> Result<Self, String>;
}

macro_rules! native {
    ($($native:ty => $variant:ident, $input:ident;)*) => {$ (
        impl ReadValue for $native {
            fn numeric(self) -> NumericScalar { NumericScalar::$variant(self) }
        }
        impl FixedValue for $native {
            const INPUT: InputType = InputType::$input;
            fn decode(bytes: &[u8], endianness: Endianness) -> Result<Self, String> {
                match decode_numeric_scalar(bytes, Self::INPUT, endianness)? {
                    NumericScalar::$variant(value) => Ok(value),
                    _ => unreachable!("selected decoder preserves native source class"),
                }
            }
        }
    )*};
}
native! {
    u8 => U8, UInt8; i8 => I8, Int8;
    u16 => U16, UInt16; i16 => I16, Int16;
    u32 => U32, UInt32; i32 => I32, Int32;
    u64 => U64, UInt64; i64 => I64, Int64;
    f32 => F32, Float32; f64 => F64, Float64;
}

fn numeric_character(value: NumericScalar) -> Result<Utf16CodeUnit, String> {
    let code = match value {
        NumericScalar::F64(value) => floating_char_code(value)?,
        NumericScalar::F32(value) => floating_char_code(f64::from(value))?,
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_u64())
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| {
                "fread: character value is outside the UTF-16 code-unit range".to_string()
            })?,
    };
    Ok(Utf16CodeUnit::from_raw(code))
}

pub(super) fn convert(value: NumericScalar, dtype: NumericDType) -> NumericScalar {
    let prototype = match dtype {
        NumericDType::F64 => return NumericScalar::F64(value.materialize_f64()),
        NumericDType::F32 => return NumericScalar::F32(value.materialize_f32()),
        NumericDType::I8 => IntegerStorage::I8(Vec::new()),
        NumericDType::I16 => IntegerStorage::I16(Vec::new()),
        NumericDType::I32 => IntegerStorage::I32(Vec::new()),
        NumericDType::I64 => IntegerStorage::I64(Vec::new()),
        NumericDType::U8 => IntegerStorage::U8(Vec::new()),
        NumericDType::U16 => IntegerStorage::U16(Vec::new()),
        NumericDType::U32 => IntegerStorage::U32(Vec::new()),
        NumericDType::U64 => IntegerStorage::U64(Vec::new()),
    };
    NumericScalar::from(match value {
        NumericScalar::F64(value) => prototype.cast_f64_assignment(value),
        NumericScalar::F32(value) => prototype.cast_f64_assignment(f64::from(value)),
        value => prototype.cast_exact_assignment(&value.into_int_value().expect("native integer")),
    })
}

pub(super) fn nonzero(value: NumericScalar) -> u8 {
    u8::from(match value {
        NumericScalar::F64(value) => value != 0.0,
        NumericScalar::F32(value) => value != 0.0,
        value => !value.into_int_value().expect("native integer").is_zero(),
    })
}
