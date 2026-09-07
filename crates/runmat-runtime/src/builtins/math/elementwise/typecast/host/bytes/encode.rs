use runmat_builtins::{TYPECAST_ERROR_INTERNAL, TYPECAST_ERROR_INVALID_INPUT};
use runmat_value::{ComplexStorage, IntValue, NumericStorage, Value};

use crate::BuiltinResult;

use super::super::super::error;

pub(super) fn value(source: Value) -> BuiltinResult<Vec<u8>> {
    match source {
        Value::Num(value) => Ok(value.to_ne_bytes().to_vec()),
        Value::Int(value) => Ok(integer(value)),
        Value::Bool(value) => Ok(vec![u8::from(value)]),
        Value::Complex(real, imag) => {
            let mut bytes = real.to_ne_bytes().to_vec();
            bytes.extend_from_slice(&imag.to_ne_bytes());
            Ok(bytes)
        }
        Value::Tensor(tensor) => tensor
            .into_numeric_storage()
            .map(numeric)
            .map_err(|cause| error::build(&TYPECAST_ERROR_INTERNAL, cause)),
        Value::LogicalArray(array) => Ok(array.data.into_vec()),
        Value::ComplexTensor(tensor) => Ok(complex(tensor.into_complex_storage())),
        _ => Err(error::build(
            &TYPECAST_ERROR_INVALID_INPUT,
            "input must be numeric or logical",
        )),
    }
}

pub(super) fn integer(value: IntValue) -> Vec<u8> {
    macro_rules! bytes {
        ($value:expr) => {
            $value.to_ne_bytes().to_vec()
        };
    }
    match value {
        IntValue::I8(value) => bytes!(value),
        IntValue::I16(value) => bytes!(value),
        IntValue::I32(value) => bytes!(value),
        IntValue::I64(value) => bytes!(value),
        IntValue::U8(value) => bytes!(value),
        IntValue::U16(value) => bytes!(value),
        IntValue::U32(value) => bytes!(value),
        IntValue::U64(value) => bytes!(value),
    }
}

fn numeric(storage: NumericStorage) -> Vec<u8> {
    macro_rules! encode {
        ($values:expr, $ty:ty) => {{
            let values = $values;
            let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<$ty>());
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            bytes
        }};
    }
    match storage {
        NumericStorage::F64(values) => encode!(values, f64),
        NumericStorage::F32(values) => encode!(values, f32),
        NumericStorage::I8(values) => values.into_iter().map(|value| value as u8).collect(),
        NumericStorage::I16(values) => encode!(values, i16),
        NumericStorage::I32(values) => encode!(values, i32),
        NumericStorage::I64(values) => encode!(values, i64),
        NumericStorage::U8(values) => values,
        NumericStorage::U16(values) => encode!(values, u16),
        NumericStorage::U32(values) => encode!(values, u32),
        NumericStorage::U64(values) => encode!(values, u64),
    }
}

fn complex(storage: ComplexStorage) -> Vec<u8> {
    match storage {
        ComplexStorage::F64(values) => values
            .into_iter()
            .flat_map(|(real, imag)| real.to_ne_bytes().into_iter().chain(imag.to_ne_bytes()))
            .collect(),
        ComplexStorage::F32(values) => values
            .into_iter()
            .flat_map(|(real, imag)| real.to_ne_bytes().into_iter().chain(imag.to_ne_bytes()))
            .collect(),
        ComplexStorage::Integer(values) => {
            let mut bytes = Vec::new();
            for index in 0..values.len() {
                bytes.extend(integer(
                    values.real.value_at(index).expect("validated real lane"),
                ));
                bytes.extend(integer(
                    values
                        .imag
                        .value_at(index)
                        .expect("validated imaginary lane"),
                ));
            }
            bytes
        }
    }
}
