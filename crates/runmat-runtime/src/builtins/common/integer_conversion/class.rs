use runmat_types::IntegerClass;
use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor};

use super::storage::integer_values;

pub(crate) trait IntegerClassExt: Sized {
    fn cast_i128(self, value: i128) -> IntValue;
    fn uses_extended_scalar_precision(self) -> bool;
    fn from_int_value(value: &IntValue) -> Self;
    fn from_storage(storage: &IntegerStorage) -> Self;
    fn accelerator_type(self) -> runmat_accelerate_api::IntegerElementType;
    fn cast_scalar(self, value: f64) -> IntValue;
    fn cast_int(self, value: &IntValue) -> IntValue;
    fn cast_tensor(self, tensor: Tensor) -> Result<Tensor, String>;
    fn storage(self, values: Vec<IntValue>) -> IntegerStorage;
}

impl IntegerClassExt for IntegerClass {
    fn cast_i128(self, value: i128) -> IntValue {
        match self {
            Self::Int8 => IntValue::I8(value.clamp(i8::MIN as i128, i8::MAX as i128) as i8),
            Self::Int16 => IntValue::I16(value.clamp(i16::MIN as i128, i16::MAX as i128) as i16),
            Self::Int32 => IntValue::I32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32),
            Self::Int64 => IntValue::I64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64),
            Self::UInt8 => IntValue::U8(value.clamp(0, u8::MAX as i128) as u8),
            Self::UInt16 => IntValue::U16(value.clamp(0, u16::MAX as i128) as u16),
            Self::UInt32 => IntValue::U32(value.clamp(0, u32::MAX as i128) as u32),
            Self::UInt64 => IntValue::U64(value.clamp(0, u64::MAX as i128) as u64),
        }
    }

    fn uses_extended_scalar_precision(self) -> bool {
        matches!(self, Self::Int64 | Self::UInt64)
    }

    fn from_int_value(value: &IntValue) -> Self {
        value.integer_class()
    }

    fn from_storage(storage: &IntegerStorage) -> Self {
        storage.integer_class()
    }

    fn accelerator_type(self) -> runmat_accelerate_api::IntegerElementType {
        self.into()
    }

    fn cast_scalar(self, value: f64) -> IntValue {
        match self {
            Self::Int8 => IntValue::I8(cast_signed(value, i8::MIN as f64, i8::MAX as f64) as i8),
            Self::Int16 => {
                IntValue::I16(cast_signed(value, i16::MIN as f64, i16::MAX as f64) as i16)
            }
            Self::Int32 => {
                IntValue::I32(cast_signed(value, i32::MIN as f64, i32::MAX as f64) as i32)
            }
            Self::Int64 => IntValue::I64(cast_signed(value, i64::MIN as f64, i64::MAX as f64)),
            Self::UInt8 => IntValue::U8(cast_unsigned(value, u8::MAX as f64) as u8),
            Self::UInt16 => IntValue::U16(cast_unsigned(value, u16::MAX as f64) as u16),
            Self::UInt32 => IntValue::U32(cast_unsigned(value, u32::MAX as f64) as u32),
            Self::UInt64 => IntValue::U64(cast_unsigned(value, u64::MAX as f64)),
        }
    }

    fn cast_int(self, value: &IntValue) -> IntValue {
        match self {
            Self::Int8 => IntValue::I8(value.to_i64().clamp(i8::MIN as i64, i8::MAX as i64) as i8),
            Self::Int16 => {
                IntValue::I16(value.to_i64().clamp(i16::MIN as i64, i16::MAX as i64) as i16)
            }
            Self::Int32 => {
                IntValue::I32(value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            }
            Self::Int64 => IntValue::I64(value.to_i64()),
            Self::UInt8 => IntValue::U8(unsigned_value(value).min(u8::MAX as u64) as u8),
            Self::UInt16 => IntValue::U16(unsigned_value(value).min(u16::MAX as u64) as u16),
            Self::UInt32 => IntValue::U32(unsigned_value(value).min(u32::MAX as u64) as u32),
            Self::UInt64 => IntValue::U64(unsigned_value(value)),
        }
    }

    fn cast_tensor(self, tensor: Tensor) -> Result<Tensor, String> {
        let shape = tensor.shape.clone();
        let storage = tensor.into_numeric_storage()?;
        let values = match storage {
            NumericStorage::F64(values) => values
                .iter()
                .map(|&value| self.cast_scalar(value))
                .collect(),
            NumericStorage::F32(values) => values
                .iter()
                .map(|&value| self.cast_scalar(f64::from(value)))
                .collect(),
            storage => {
                let storage = storage.into_integer_storage().map_err(|_| {
                    "unsupported numeric storage for integer conversion".to_string()
                })?;
                integer_values(storage)
                    .iter()
                    .map(|value| self.cast_int(value))
                    .collect()
            }
        };
        Tensor::new_integer(self.storage(values), shape)
    }

    fn storage(self, values: Vec<IntValue>) -> IntegerStorage {
        match self {
            Self::Int8 => IntegerStorage::I8(
                values
                    .iter()
                    .map(|value| value.to_i64().clamp(i8::MIN as i64, i8::MAX as i64) as i8)
                    .collect(),
            ),
            Self::Int16 => IntegerStorage::I16(
                values
                    .iter()
                    .map(|value| value.to_i64().clamp(i16::MIN as i64, i16::MAX as i64) as i16)
                    .collect(),
            ),
            Self::Int32 => IntegerStorage::I32(
                values
                    .iter()
                    .map(|value| value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                    .collect(),
            ),
            Self::Int64 => IntegerStorage::I64(values.iter().map(IntValue::to_i64).collect()),
            Self::UInt8 => IntegerStorage::U8(
                values
                    .iter()
                    .map(|value| unsigned_value(value).min(u8::MAX as u64) as u8)
                    .collect(),
            ),
            Self::UInt16 => IntegerStorage::U16(
                values
                    .iter()
                    .map(|value| unsigned_value(value).min(u16::MAX as u64) as u16)
                    .collect(),
            ),
            Self::UInt32 => IntegerStorage::U32(
                values
                    .iter()
                    .map(|value| unsigned_value(value).min(u32::MAX as u64) as u32)
                    .collect(),
            ),
            Self::UInt64 => IntegerStorage::U64(values.iter().map(unsigned_value).collect()),
        }
    }
}

fn cast_signed(value: f64, min: f64, max: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            min as i64
        } else {
            max as i64
        }
    } else {
        value.round().clamp(min, max) as i64
    }
}

fn cast_unsigned(value: f64, max: f64) -> u64 {
    if value.is_nan() || value.is_sign_negative() {
        0
    } else if value.is_infinite() {
        max as u64
    } else {
        value.round().clamp(0.0, max) as u64
    }
}

fn unsigned_value(value: &IntValue) -> u64 {
    match value {
        IntValue::U64(value) => *value,
        _ => value.to_i64().max(0) as u64,
    }
}
